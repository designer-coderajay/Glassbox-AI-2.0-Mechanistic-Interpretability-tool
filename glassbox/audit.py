"""
glassbox/audit.py — Black-Box Audit Mode
=========================================

Audits ANY AI model via its API — no model weights, no TransformerLens needed.
Works on GPT-4, Claude, Llama, Gemini, or any proprietary LLM via HTTP.

This is the completion of Glassbox's 100% market coverage strategy:
    White-box mode (core.py): TransformerLens open-source models
    Black-box mode (audit.py): ANY model via API — this file

Black-box methodology
---------------------
Without access to model internals, we cannot do attribution patching.
Instead, we use behavioural probing — a structured battery of tests that
reveals what the model knows, what it ignores, and how it makes decisions.

Four pillars of black-box explainability:
1. Counterfactual probing   — flip one variable, measure output change
2. Sensitivity analysis     — which input tokens drive the prediction
3. Consistency testing      — same question rephrased, same answer?
4. Faithfulness estimation  — does the model's stated reasoning match its output?

These map to EU AI Act Article 13 requirements:
    - Sufficiency proxy:       sensitivity_score (how much variation we can explain)
    - Comprehensiveness proxy: consistency_score (do identified drivers always predict)
    - F1 proxy:                combined_score
    - Article 13(3)(b):        full behavioural probe report

The black-box report integrates with AnnexIVReport exactly like white-box results.

Usage
-----
    from glassbox.audit import BlackBoxAuditor, ModelProvider

    # Works with any API that accepts {"messages": [...]} and returns {"content": str}
    auditor = BlackBoxAuditor(
        model_provider = ModelProvider.OPENAI,
        model_name     = "gpt-4",
        api_key        = "sk-...",
    )

    result = auditor.audit(
        decision_prompt    = "The loan applicant has a credit score of 620 and annual income of $45,000. Based on standard banking criteria, the loan application should be",
        expected_positive  = "approved",
        expected_negative  = "denied",
        context_variables  = {
            "credit_score": 620,
            "annual_income": 45000,
            "loan_amount":  25000,
        },
    )

    print(result["black_box_faithfulness"]["combined_score"])

    # Drop-in with AnnexIVReport
    from glassbox.compliance import AnnexIVReport, DeploymentContext
    report = AnnexIVReport(...)
    report.add_analysis(result, use_case="Loan approval decision")
    report.to_pdf("compliance_report.pdf")
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "BlackBoxAuditor",
    "ModelProvider",
    "BlackBoxResult",
    "CounterfactualProbe",
    "SensitivityProbe",
]

# ---------------------------------------------------------------------------
# Supported providers
# ---------------------------------------------------------------------------

class ModelProvider(str, Enum):
    """
    API-compatible AI providers.
    Any provider can also be reached with CUSTOM + a custom http_fn.
    """
    OPENAI     = "openai"        # api.openai.com
    ANTHROPIC  = "anthropic"     # api.anthropic.com
    TOGETHER   = "together"      # api.together.xyz  (Llama, Mistral, etc.)
    GROQ       = "groq"          # api.groq.com      (Llama, Mixtral)
    AZURE      = "azure"         # Azure OpenAI Service
    CUSTOM     = "custom"        # Supply your own http_fn


# ---------------------------------------------------------------------------
# Probe structures
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualProbe:
    """
    A single counterfactual test: change one variable, observe output change.
    Maps to EU AI Act Article 13 — understanding what drives decisions.
    """
    variable_name:    str
    original_value:   Any
    counterfactual_value: Any
    original_prompt:  str
    cf_prompt:        str
    original_output:  str = ""
    cf_output:        str = ""
    output_changed:   bool = False
    causal_impact:    float = 0.0  # 0 = no impact, 1 = full reversal


@dataclass
class SensitivityProbe:
    """
    Tests how sensitive the model is to each context variable.
    Proxy for attribution patching's per-head attribution scores.
    """
    variable_name:   str
    sensitivity:     float  # 0-1, measured by output variation across value range
    direction:       str    # "positive" or "negative" (higher = more/less favourable)
    num_probes:      int    = 5


# ---------------------------------------------------------------------------
# Main result structure — compatible with AnnexIVReport.add_analysis()
# ---------------------------------------------------------------------------

@dataclass
class BlackBoxResult:
    """
    Full black-box audit result — drop-in compatible with GlassboxV2.analyze().
    The structure matches analyze()'s return dict so AnnexIVReport works unchanged.
    """
    # GlassboxV2.analyze()-compatible fields
    circuit:           List[str]          = field(default_factory=list)   # variable names
    n_heads:           int                = 0                              # n_significant_variables
    clean_ld:          float              = 0.0                           # mean confidence delta
    corr_prompt:       str                = ""
    attributions:      Dict[str, float]   = field(default_factory=dict)  # variable -> impact
    mlp_attributions:  Dict[str, float]   = field(default_factory=dict)
    top_heads:         List[Dict]         = field(default_factory=list)  # top variables
    method:            str                = "black_box_behavioural_probing"
    faithfulness:      Dict[str, Any]     = field(default_factory=dict)
    model_metadata:    Dict[str, Any]     = field(default_factory=dict)

    # Black-box specific
    counterfactual_probes: List[CounterfactualProbe] = field(default_factory=list)
    sensitivity_probes:    List[SensitivityProbe]    = field(default_factory=list)
    consistency_score:     float = 0.0
    rephrase_probes:       List[Dict] = field(default_factory=list)
    confidence_scores:     Dict[str, float] = field(default_factory=dict)
    audit_timestamp:       str = ""
    total_api_calls:       int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict compatible with AnnexIVReport.add_analysis()."""
        return {
            "circuit":           self.circuit,
            "n_heads":           self.n_heads,
            "clean_ld":          self.clean_ld,
            "corr_prompt":       self.corr_prompt,
            "attributions":      self.attributions,
            "mlp_attributions":  self.mlp_attributions,
            "top_heads":         self.top_heads,
            "method":            self.method,
            "faithfulness":      self.faithfulness,
            "model_metadata":    self.model_metadata,
            # black-box extras
            "black_box_faithfulness": {
                "sensitivity_score": self.faithfulness.get("sufficiency", 0),
                "consistency_score": self.consistency_score,
                "combined_score":    self.faithfulness.get("f1", 0),
            },
            "counterfactual_probes": [
                {
                    "variable":      p.variable_name,
                    "original":      str(p.original_value),
                    "counterfactual": str(p.counterfactual_value),
                    "output_changed": p.output_changed,
                    "causal_impact": p.causal_impact,
                }
                for p in self.counterfactual_probes
            ],
            "sensitivity_by_variable": {
                p.variable_name: {"sensitivity": p.sensitivity, "direction": p.direction}
                for p in self.sensitivity_probes
            },
            "total_api_calls": self.total_api_calls,
            "audit_timestamp": self.audit_timestamp,
        }


# ---------------------------------------------------------------------------
# Core auditor
# ---------------------------------------------------------------------------

class BlackBoxAuditor:
    """
    Audits any AI model via its API using behavioural probing.

    Works with OpenAI, Anthropic, Together, Groq, Azure OpenAI,
    or any custom HTTP endpoint.

    Parameters
    ----------
    model_provider : ModelProvider — which API provider to use
    model_name     : str — model identifier (e.g. "gpt-4", "claude-3-opus-20240229")
    api_key        : str — API key (or set via environment variable)
    base_url       : str — custom base URL (required for AZURE/CUSTOM)
    http_fn        : callable — for CUSTOM provider, pass your own function:
                     http_fn(prompt: str, system: str) -> str
    max_tokens     : int — max tokens in model response (default 512)
    temperature    : float — sampling temperature (default 0.0 for determinism)
    rate_limit_rps : float — requests per second limit (default 2.0)
    verbose        : bool — log each probe call (default False)

    Example
    -------
        auditor = BlackBoxAuditor(
            model_provider = ModelProvider.OPENAI,
            model_name     = "gpt-4",
            api_key        = os.environ["OPENAI_API_KEY"],
        )
    """

    _OPENAI_BASE    = "https://api.openai.com/v1/chat/completions"
    _ANTHROPIC_BASE = "https://api.anthropic.com/v1/messages"
    _TOGETHER_BASE  = "https://api.together.xyz/v1/chat/completions"
    _GROQ_BASE      = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model_provider:  ModelProvider = ModelProvider.OPENAI,
        model_name:      str = "gpt-4",
        api_key:         str = "",
        base_url:        str = "",
        http_fn:         Optional[Callable[[str, str], str]] = None,
        max_tokens:      int   = 512,
        temperature:     float = 0.0,
        rate_limit_rps:  float = 2.0,
        verbose:         bool  = False,
    ):
        self.model_provider  = model_provider
        self.model_name      = model_name
        self.api_key         = api_key
        self.base_url        = base_url
        self.http_fn         = http_fn
        self.max_tokens      = max_tokens
        self.temperature     = temperature
        self._min_interval   = 1.0 / max(rate_limit_rps, 0.1)
        self._last_call_time = 0.0
        self.verbose         = verbose
        self._call_count     = 0

        if model_provider == ModelProvider.CUSTOM and http_fn is None:
            raise ValueError(
                "ModelProvider.CUSTOM requires an http_fn parameter. "
                "Provide a callable: http_fn(prompt: str, system: str) -> str"
            )

    # ------------------------------------------------------------------
    # Primary audit method
    # ------------------------------------------------------------------

    def audit(
        self,
        decision_prompt:    str,
        expected_positive:  str,
        expected_negative:  str,
        context_variables:  Optional[Dict[str, Any]] = None,
        n_rephrases:        int = 3,
        n_sensitivity_steps: int = 5,
        system_prompt:      str = "You are a decision-making AI system. Respond concisely.",
    ) -> Dict[str, Any]:
        """
        Run a full black-box audit of the AI model.

        Parameters
        ----------
        decision_prompt    : The prompt that leads to a binary decision
        expected_positive  : The "positive/approval" output token (e.g. "approved")
        expected_negative  : The "denial/rejection" output token (e.g. "denied")
        context_variables  : Dict of variable_name -> value. Used for counterfactual
                             and sensitivity probes. E.g. {"credit_score": 620}
        n_rephrases        : Number of rephrased prompts to test for consistency
        n_sensitivity_steps: Steps per variable for sensitivity analysis
        system_prompt      : System-level instruction sent with each probe

        Returns
        -------
        dict — compatible with AnnexIVReport.add_analysis() and GlassboxV2.analyze()
        """
        self._call_count = 0
        start_time = time.time()

        result = BlackBoxResult()
        result.corr_prompt = decision_prompt
        result.model_metadata = {
            "model_name":       self.model_name,
            "provider":         self.model_provider.value,
            "n_layers":         None,  # not available in black-box mode
            "n_heads":          None,
            "d_model":          None,
            "d_head":           None,
            "glassbox_version": self._get_version(),
        }

        # 1. Baseline — what does the model say on the clean prompt?
        baseline_output = self._call(decision_prompt, system_prompt)
        baseline_favours_positive = self._output_favours(
            baseline_output, expected_positive, expected_negative
        )
        logger.info("Baseline output: %s (favours_positive=%s)", baseline_output[:80], baseline_favours_positive)

        # 2. Counterfactual probes — one variable at a time
        cf_probes = []
        if context_variables:
            cf_probes = self._run_counterfactual_probes(
                decision_prompt, expected_positive, expected_negative,
                context_variables, system_prompt,
            )
        result.counterfactual_probes = cf_probes

        # 3. Sensitivity analysis — sweep variable values
        sens_probes = []
        if context_variables:
            sens_probes = self._run_sensitivity_probes(
                decision_prompt, expected_positive, expected_negative,
                context_variables, system_prompt, n_sensitivity_steps,
            )
        result.sensitivity_probes = sens_probes

        # 4. Consistency probes — rephrase the same question
        consistency_score, rephrase_results = self._run_consistency_probes(
            decision_prompt, expected_positive, expected_negative,
            baseline_favours_positive, system_prompt, n_rephrases,
        )
        result.consistency_score = consistency_score
        result.rephrase_probes   = rephrase_results

        # 5. Compute aggregated attribution-like scores
        attributions = {}
        if cf_probes:
            for p in cf_probes:
                attributions[p.variable_name] = p.causal_impact
        result.attributions = attributions

        # Sensitivity as mlp_attributions (structural importance)
        if sens_probes:
            result.mlp_attributions = {p.variable_name: p.sensitivity for p in sens_probes}

        # Circuit = variables with high causal impact
        significant_vars = sorted(
            [k for k, v in attributions.items() if v > 0.3],
            key=lambda k: -attributions[k],
        )
        result.circuit = significant_vars if significant_vars else list(attributions.keys())[:3]
        result.n_heads = len(result.circuit)

        # Top heads — the most impactful variables (analogous to attention heads)
        sorted_attrs = sorted(attributions.items(), key=lambda kv: -kv[1])
        result.top_heads = [
            {
                "layer": 0,  # not applicable for black-box
                "head":  0,
                "variable": k,
                "attr":  v,
                "rel_depth": 0.0,
            }
            for k, v in sorted_attrs[:10]
        ]

        # Average causal impact as proxy for logit difference
        result.clean_ld = sum(attributions.values()) / len(attributions) if attributions else 0.5

        # 6. Compute faithfulness metrics (behavioural proxies)
        sensitivity_score = (
            sum(p.sensitivity for p in sens_probes) / len(sens_probes)
            if sens_probes else
            sum(p.causal_impact for p in cf_probes) / len(cf_probes) if cf_probes else 0.5
        )
        comp_proxy = consistency_score
        f1 = 2 * sensitivity_score * comp_proxy / (sensitivity_score + comp_proxy) \
             if (sensitivity_score + comp_proxy) > 0 else 0.0

        category = _categorise_faithfulness(sensitivity_score, comp_proxy, f1)
        result.faithfulness = {
            "sufficiency":         sensitivity_score,   # sensitivity proxy
            "comprehensiveness":   comp_proxy,          # consistency proxy
            "f1":                  f1,
            "category":            category,
            "suff_is_approx":      True,  # behavioural proxy, not exact causal
        }

        elapsed = time.time() - start_time
        result.total_api_calls = self._call_count
        result.audit_timestamp = _iso_now()

        logger.info(
            "Black-box audit complete: %d API calls, %.1fs, F1=%.3f",
            self._call_count, elapsed, f1,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Probe runners
    # ------------------------------------------------------------------

    def _run_counterfactual_probes(
        self,
        base_prompt:       str,
        expected_positive: str,
        expected_negative: str,
        variables:         Dict[str, Any],
        system_prompt:     str,
    ) -> List[CounterfactualProbe]:
        """
        For each variable, substitute its value and measure output change.
        E.g. credit_score=620 -> credit_score=800 — does the decision flip?
        """
        probes = []
        base_output = self._call(base_prompt, system_prompt)
        base_favours = self._output_favours(base_output, expected_positive, expected_negative)

        for var_name, original_val in variables.items():
            cf_val = self._generate_counterfactual_value(var_name, original_val)
            if cf_val is None:
                continue

            cf_prompt = base_prompt.replace(str(original_val), str(cf_val))
            if cf_prompt == base_prompt:
                # Value not found in prompt — skip
                continue

            cf_output  = self._call(cf_prompt, system_prompt)
            cf_favours = self._output_favours(cf_output, expected_positive, expected_negative)

            output_changed = (base_favours != cf_favours)
            # Causal impact: 1.0 if decision flipped, 0.5 if changed but didn't flip, 0.0 unchanged
            causal_impact = 1.0 if output_changed else (
                0.5 if self._output_significantly_different(base_output, cf_output) else 0.0
            )

            probe = CounterfactualProbe(
                variable_name        = var_name,
                original_value       = original_val,
                counterfactual_value = cf_val,
                original_prompt      = base_prompt,
                cf_prompt            = cf_prompt,
                original_output      = base_output,
                cf_output            = cf_output,
                output_changed       = output_changed,
                causal_impact        = causal_impact,
            )
            probes.append(probe)
            logger.debug("CF probe %s: %s->%s, changed=%s, impact=%.2f",
                         var_name, original_val, cf_val, output_changed, causal_impact)

        return probes

    def _run_sensitivity_probes(
        self,
        base_prompt:       str,
        expected_positive: str,
        expected_negative: str,
        variables:         Dict[str, Any],
        system_prompt:     str,
        n_steps:           int,
    ) -> List[SensitivityProbe]:
        """
        Sweep each variable across a range of values.
        Measures what fraction of values cause a direction change.
        """
        probes = []
        for var_name, original_val in variables.items():
            sweep_values = self._generate_value_sweep(var_name, original_val, n_steps)
            if not sweep_values:
                continue

            positive_count = 0
            negative_count = 0
            for val in sweep_values:
                test_prompt = base_prompt.replace(str(original_val), str(val))
                if test_prompt == base_prompt:
                    continue
                output    = self._call(test_prompt, system_prompt)
                favours   = self._output_favours(output, expected_positive, expected_negative)
                if favours:
                    positive_count += 1
                else:
                    negative_count += 1

            total = positive_count + negative_count
            if total == 0:
                continue

            # Sensitivity = how much the variable moved decisions
            # High variance = high sensitivity
            p_pos = positive_count / total
            sensitivity = 2 * p_pos * (1 - p_pos) * 2  # peaks at 0.5/0.5 split = 1.0
            sensitivity = min(1.0, sensitivity)
            direction   = "positive" if positive_count > negative_count else "negative"

            probes.append(SensitivityProbe(
                variable_name = var_name,
                sensitivity   = sensitivity,
                direction     = direction,
                num_probes    = total,
            ))

        return probes

    def _run_consistency_probes(
        self,
        base_prompt:        str,
        expected_positive:  str,
        expected_negative:  str,
        baseline_positive:  bool,
        system_prompt:      str,
        n_rephrases:        int,
    ) -> Tuple[float, List[Dict]]:
        """
        Rephrase the base prompt n times. Measure fraction of outputs
        that agree with the baseline. High consistency = comprehensiveness proxy.
        """
        if n_rephrases == 0:
            return 1.0, []

        rephrase_prompts = self._generate_rephrases(base_prompt, n_rephrases)
        results = []
        agree_count = 0

        for i, rephrase in enumerate(rephrase_prompts):
            output  = self._call(rephrase, system_prompt)
            favours = self._output_favours(output, expected_positive, expected_negative)
            agrees  = (favours == baseline_positive)
            if agrees:
                agree_count += 1
            results.append({
                "rephrase_n":      i + 1,
                "prompt":          rephrase,
                "output":          output[:200],
                "favours_positive": favours,
                "agrees_with_baseline": agrees,
            })

        consistency = agree_count / len(rephrase_prompts) if rephrase_prompts else 1.0
        return consistency, results

    # ------------------------------------------------------------------
    # HTTP call layer
    # ------------------------------------------------------------------

    def _call(self, prompt: str, system: str = "") -> str:
        """
        Call the model API with rate limiting. Returns response text.
        """
        # Rate limit
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        self._last_call_time = time.time()
        self._call_count += 1

        if self.verbose:
            logger.info("[Call %d] Prompt[:80]: %s", self._call_count, prompt[:80])

        if self.model_provider == ModelProvider.CUSTOM and self.http_fn:
            return self.http_fn(prompt, system)

        dispatcher = {
            ModelProvider.OPENAI:    self._call_openai_compatible,
            ModelProvider.TOGETHER:  self._call_openai_compatible,
            ModelProvider.GROQ:      self._call_openai_compatible,
            ModelProvider.AZURE:     self._call_openai_compatible,
            ModelProvider.ANTHROPIC: self._call_anthropic,
        }
        fn = dispatcher.get(self.model_provider, self._call_openai_compatible)
        return fn(prompt, system)

    def _call_openai_compatible(self, prompt: str, system: str) -> str:
        """OpenAI-compatible API call (OpenAI, Together, Groq, Azure)."""
        base = {
            ModelProvider.OPENAI:   self._OPENAI_BASE,
            ModelProvider.TOGETHER: self._TOGETHER_BASE,
            ModelProvider.GROQ:     self._GROQ_BASE,
        }.get(self.model_provider, self.base_url or self._OPENAI_BASE)
        if self.base_url:
            base = self.base_url

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model":       self.model_name,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }).encode()

        req = urllib.request.Request(
            base,
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"API error {e.code} from {self.model_provider.value}: {body[:200]}"
            ) from e

    def _call_anthropic(self, prompt: str, system: str) -> str:
        """Anthropic Messages API call."""
        payload = json.dumps({
            "model":      self.model_name,
            "max_tokens": self.max_tokens,
            "system":     system or "You are a helpful assistant.",
            "messages":   [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            self._ANTHROPIC_BASE,
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(
                f"Anthropic API error {e.code}: {body[:200]}"
            ) from e

    # ------------------------------------------------------------------
    # Helpers: counterfactual value generation
    # ------------------------------------------------------------------

    def _generate_counterfactual_value(self, var_name: str, original: Any) -> Optional[Any]:
        """
        Generate a meaningful counterfactual value for a given variable.
        Heuristic: nudge numeric values significantly, flip booleans, swap categories.
        """
        name_lower = var_name.lower()

        if isinstance(original, bool):
            return not original

        if isinstance(original, (int, float)):
            # For scores/amounts, flip between high and low
            if "score" in name_lower or "rating" in name_lower:
                # credit score: if low (<660) -> high (780), if high -> low
                return 780 if original < 660 else 480
            if "income" in name_lower or "salary" in name_lower or "revenue" in name_lower:
                return original * 3 if original < 50000 else original // 3
            if "debt" in name_lower or "loan" in name_lower or "amount" in name_lower:
                return original * 2 if original < 100000 else original // 2
            if "age" in name_lower:
                return 45 if original < 35 else 28
            # Generic numeric: flip 20% / 80% of range
            return int(original * 2.5) if original < 1000 else int(original * 0.4)

        if isinstance(original, str):
            # Category flip heuristics
            low = original.lower()
            if low in ("yes", "true", "approved", "accepted"):
                return "no"
            if low in ("no", "false", "denied", "rejected"):
                return "yes"
            if low in ("good", "excellent", "high"):
                return "poor"
            if low in ("poor", "bad", "low"):
                return "good"
            return None  # can't generate useful counterfactual for arbitrary strings

        return None

    def _generate_value_sweep(self, var_name: str, original: Any, n: int) -> List[Any]:
        """Generate n values spanning the range of a variable for sensitivity analysis."""
        if not isinstance(original, (int, float)):
            return []

        name_lower = var_name.lower()
        if "score" in name_lower or "rating" in name_lower:
            lo, hi = 300, 850
        elif "income" in name_lower or "salary" in name_lower:
            lo, hi = max(10000, original // 5), original * 5
        elif "age" in name_lower:
            lo, hi = 21, 70
        elif "amount" in name_lower or "loan" in name_lower:
            lo, hi = max(1000, original // 5), original * 5
        else:
            lo, hi = max(0, original // 5), original * 5

        step = (hi - lo) / max(n - 1, 1)
        return [int(lo + i * step) for i in range(n)]

    def _generate_rephrases(self, prompt: str, n: int) -> List[str]:
        """
        Generate rephrased versions of the prompt.
        Uses template transformations — no extra API calls needed.
        """
        rephrases = []

        # Transform 1: passive/active voice flip
        r1 = prompt.replace("should be", "is likely to be")
        if r1 != prompt:
            rephrases.append(r1)

        # Transform 2: question form
        r2 = "Considering the following information: " + prompt.replace("should be", "— what is the recommendation?")
        rephrases.append(r2)

        # Transform 3: formal framing
        r3 = "As a financial risk analyst, evaluate: " + prompt
        rephrases.append(r3)

        # Transform 4: enumerate phrasing
        lines = prompt.split(",")
        if len(lines) > 2:
            r4 = "Given: " + "; ".join(l.strip() for l in lines) + ". What is the decision?"
            rephrases.append(r4)

        # Transform 5: minimal phrasing
        r5 = "Decision: " + prompt[:150]
        rephrases.append(r5)

        return rephrases[:n]

    # ------------------------------------------------------------------
    # Helpers: output parsing
    # ------------------------------------------------------------------

    def _output_favours(self, output: str, positive: str, negative: str) -> bool:
        """
        Determine if model output favours the positive or negative outcome.
        Returns True if positive is indicated, False if negative or ambiguous.
        """
        out_lower = output.lower()
        pos_lower = positive.lower()
        neg_lower = negative.lower()

        pos_count = out_lower.count(pos_lower)
        neg_count = out_lower.count(neg_lower)

        if pos_count > neg_count:
            return True
        if neg_count > pos_count:
            return False

        # Tie-breaking: check for sentiment words near start of response
        first_200 = out_lower[:200]
        positive_signals = {"approv", "accept", "grant", "yes", "eligible", "qualify", "recommend"}
        negative_signals = {"den", "reject", "declin", "not eligible", "does not qualify", "unable"}

        pos_signals = sum(1 for s in positive_signals if s in first_200)
        neg_signals = sum(1 for s in negative_signals if s in first_200)
        return pos_signals >= neg_signals

    def _output_significantly_different(self, output_a: str, output_b: str) -> bool:
        """
        Rough string distance check — are outputs meaningfully different?
        True if overlap < 80% of shorter output.
        """
        words_a = set(output_a.lower().split())
        words_b = set(output_b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        min_len = min(len(words_a), len(words_b))
        return (overlap / min_len) < 0.80

    def _get_version(self) -> str:
        try:
            from glassbox import __version__
            return __version__
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# Public helper: create BlackBoxAuditor from env variables
# ---------------------------------------------------------------------------

def from_env(
    model_provider: ModelProvider = ModelProvider.OPENAI,
    model_name:     str           = "gpt-4",
    **kwargs,
) -> "BlackBoxAuditor":
    """
    Create a BlackBoxAuditor using API keys from environment variables.

    Environment variables checked:
        OPENAI_API_KEY, ANTHROPIC_API_KEY, TOGETHER_API_KEY, GROQ_API_KEY

    Example
    -------
        auditor = from_env(ModelProvider.ANTHROPIC, "claude-3-opus-20240229")
    """
    import os
    key_map = {
        ModelProvider.OPENAI:    "OPENAI_API_KEY",
        ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        ModelProvider.TOGETHER:  "TOGETHER_API_KEY",
        ModelProvider.GROQ:      "GROQ_API_KEY",
    }
    env_var = key_map.get(model_provider, "API_KEY")
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise EnvironmentError(
            f"API key not found. Set the {env_var} environment variable."
        )
    return BlackBoxAuditor(
        model_provider=model_provider,
        model_name=model_name,
        api_key=api_key,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _categorise_faithfulness(suff: float, comp: float, f1: float) -> str:
    if suff > 0.9 and comp < 0.4:
        return "backup_mechanisms"
    if suff > 0.7 and comp > 0.5:
        return "faithful"
    if suff < 0.5:
        return "incomplete"
    if suff < 0.6 and comp < 0.5:
        return "weak"
    return "moderate"


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
