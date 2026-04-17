# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ajay Pravin Mahale
"""
glassbox/hf_integration.py — HuggingFace Hub Integration
=========================================================

Two capabilities:

1. ``load_from_hub(repo_id)``
   Wrapper around ``HookedTransformer.from_pretrained`` that accepts any
   HuggingFace Hub repo ID (not just the short aliases TransformerLens
   knows by name). Normalises model architecture names automatically.

2. ``HuggingFaceModelCard``
   Reads and writes EU AI Act Annex IV compliance metadata directly
   into a HuggingFace Hub model card (README.md).

   Adds a structured ``## EU AI Act Compliance`` section with:
   - Glassbox analysis summary
   - Faithfulness scores
   - Circuit information
   - Compliance status badge
   - Link to full Annex IV report (if hosted)

   Writes back via the HuggingFace Hub API (requires ``huggingface_hub``
   package and a write token).

Usage
-----
::

    # 1. Load any HF Hub model directly into Glassbox
    from glassbox.hf_integration import load_from_hub
    from glassbox import GlassboxV2

    model = load_from_hub("EleutherAI/gpt-neo-125m")
    gb    = GlassboxV2(model)
    result = gb.analyze(prompt, correct, incorrect)

    # 2. Push compliance metadata to the model card
    from glassbox.hf_integration import HuggingFaceModelCard

    card = HuggingFaceModelCard("EleutherAI/gpt-neo-125m", token="hf_...")
    card.push_compliance_section(
        result,
        annex_iv_url="https://your-server.example.com/reports/ABC123.pdf",
        auditor="Ajay Mahale <mahale.ajay01@gmail.com>",
    )

    # 3. Read back compliance section from a model card
    info = card.read_compliance_section()
    print(info["compliance_status"])
    print(info["faithfulness_sufficiency"])

Requirements
------------
    pip install huggingface_hub transformer-lens
    # optional (for push):  huggingface-cli login   OR  pass token= explicitly
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Lazy imports — huggingface_hub is optional (not a hard dependency)
# ---------------------------------------------------------------------------

def _require_hf_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF Hub integration.\n"
            "Install it:  pip install huggingface_hub"
        )


def _require_tl():
    try:
        import transformer_lens
        return transformer_lens
    except ImportError:
        raise ImportError(
            "transformer_lens is required to load models.\n"
            "Install it:  pip install transformer-lens"
        )


# ---------------------------------------------------------------------------
# Architecture normalisation map
# (TransformerLens canonical name → common HF Hub repo patterns)
# ---------------------------------------------------------------------------

_ARCH_ALIASES: Dict[str, List[str]] = {
    "gpt2":           ["gpt2", "openai-gpt"],
    "gpt2-medium":    ["gpt2-medium"],
    "gpt2-large":     ["gpt2-large"],
    "gpt2-xl":        ["gpt2-xl"],
    "gpt-neo-125M":   ["EleutherAI/gpt-neo-125m", "gpt-neo-125"],
    "gpt-neo-1.3B":   ["EleutherAI/gpt-neo-1.3B"],
    "gpt-neo-2.7B":   ["EleutherAI/gpt-neo-2.7B"],
    "gpt-j-6B":       ["EleutherAI/gpt-j-6B"],
    "gpt-neox-20b":   ["EleutherAI/gpt-neox-20b"],
    "opt-125m":       ["facebook/opt-125m"],
    "opt-1.3b":       ["facebook/opt-1.3b"],
    "opt-2.7b":       ["facebook/opt-2.7b"],
    "opt-6.7b":       ["facebook/opt-6.7b"],
    "pythia-70m":     ["EleutherAI/pythia-70m"],
    "pythia-160m":    ["EleutherAI/pythia-160m"],
    "pythia-410m":    ["EleutherAI/pythia-410m"],
    "pythia-1b":      ["EleutherAI/pythia-1b"],
    "pythia-2.8b":    ["EleutherAI/pythia-2.8b"],
    "pythia-6.9b":    ["EleutherAI/pythia-6.9b"],
    "pythia-12b":     ["EleutherAI/pythia-12b"],
    "llama-7b":       ["meta-llama/Llama-2-7b-hf", "huggyllama/llama-7b"],
    "llama-13b":      ["meta-llama/Llama-2-13b-hf"],
    "llama-3-8b":     ["meta-llama/Meta-Llama-3-8B"],
    "mistral-7b":     ["mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"],
    "phi-2":          ["microsoft/phi-2"],
    "phi-3-mini":     ["microsoft/Phi-3-mini-4k-instruct"],
    "gemma-2b":       ["google/gemma-2b"],
    "gemma-7b":       ["google/gemma-7b"],
    "falcon-7b":      ["tiiuae/falcon-7b"],
}


def _resolve_tl_name(repo_id: str) -> str:
    """Map a HF Hub repo ID to the best TransformerLens canonical name."""
    repo_lower = repo_id.lower()
    for tl_name, aliases in _ARCH_ALIASES.items():
        for alias in aliases:
            if alias.lower() in repo_lower or repo_lower in alias.lower():
                return tl_name
    # Fall through — TransformerLens will try the raw string
    return repo_id


# ---------------------------------------------------------------------------
# Public function 1: load_from_hub
# ---------------------------------------------------------------------------

def load_from_hub(
    repo_id: str,
    revision: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    **tl_kwargs,
):
    """
    Load any HuggingFace Hub model into TransformerLens.

    Parameters
    ----------
    repo_id : str
        HuggingFace Hub repository ID, e.g. ``"EleutherAI/gpt-neo-125m"``.
    revision : str, optional
        Git revision (branch, tag, or commit SHA). Defaults to ``main``.
    device : str, optional
        ``'cpu'``, ``'cuda'``, or ``'mps'``. Auto-detected if not set.
    dtype : str, optional
        ``'float32'`` or ``'float16'``. Defaults to ``'float32'``.
    **tl_kwargs
        Additional keyword arguments forwarded to
        ``HookedTransformer.from_pretrained``.

    Returns
    -------
    transformer_lens.HookedTransformer
        A TransformerLens hooked transformer ready for Glassbox analysis.

    Raises
    ------
    ImportError
        If ``transformer_lens`` is not installed.
    ValueError
        If the model architecture is not supported by TransformerLens.

    Example
    -------
    ::

        model = load_from_hub("EleutherAI/gpt-neo-125m")
        gb    = GlassboxV2(model)
    """
    tl = _require_tl()
    tl_name = _resolve_tl_name(repo_id)

    kwargs: Dict[str, Any] = {
        "center_unembed":                True,
        "center_writing_weights":        True,
        "fold_ln":                       True,
        "refactor_factored_attn_matrices": True,
    }
    if revision:
        kwargs["revision"] = revision
    if dtype:
        kwargs["dtype"] = dtype
    kwargs.update(tl_kwargs)

    try:
        model = tl.HookedTransformer.from_pretrained(tl_name, **kwargs)
    except Exception as e:
        # Retry with raw repo_id if resolved name fails
        if tl_name != repo_id:
            try:
                model = tl.HookedTransformer.from_pretrained(repo_id, **kwargs)
            except Exception:
                raise ValueError(
                    f"Could not load '{repo_id}' via TransformerLens.\n"
                    f"Resolved name tried: '{tl_name}'\n"
                    f"Original error: {e}\n\n"
                    f"Supported architectures: {list(_ARCH_ALIASES.keys())}\n"
                    f"For unsupported models, use BlackBoxAuditor instead:\n"
                    f"  from glassbox import BlackBoxAuditor"
                ) from e
        else:
            raise

    if device:
        model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Public class 2: HuggingFaceModelCard
# ---------------------------------------------------------------------------

_COMPLIANCE_SECTION_HEADER = "## EU AI Act Compliance (Glassbox)"
_COMPLIANCE_JSON_START      = "<!-- glassbox:compliance:start -->"
_COMPLIANCE_JSON_END        = "<!-- glassbox:compliance:end -->"


class HuggingFaceModelCard:
    """
    Read and write EU AI Act Annex IV compliance data in a HuggingFace Hub
    model card (README.md).

    Parameters
    ----------
    repo_id : str
        HuggingFace Hub repository ID.
    token : str, optional
        HuggingFace Hub write token. If not provided, uses the token cached
        by ``huggingface-cli login``.
    repo_type : str
        ``'model'`` (default) or ``'dataset'``.
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: str = "model",
    ) -> None:
        self._hf = _require_hf_hub()
        self.repo_id   = repo_id
        self.token     = token
        self.repo_type = repo_type

    # ------------------------------------------------------------------
    # Push compliance section
    # ------------------------------------------------------------------

    def push_compliance_section(
        self,
        result: Dict[str, Any],
        annex_iv_url: Optional[str] = None,
        auditor: Optional[str] = None,
        commit_message: str = "chore: add Glassbox EU AI Act compliance metadata",
    ) -> str:
        """
        Add or update the ``## EU AI Act Compliance`` section in the model card.

        Parameters
        ----------
        result : dict
            Output of ``GlassboxV2.analyze()``.
        annex_iv_url : str, optional
            URL to the full hosted Annex IV PDF report.
        auditor : str, optional
            Name / email of the person running the audit.
        commit_message : str
            Git commit message for the model card update.

        Returns
        -------
        str
            The URL of the updated model card on HuggingFace Hub.
        """
        current_card = self._fetch_card_content()
        new_section  = self._build_section(result, annex_iv_url, auditor)
        updated_card = self._inject_section(current_card, new_section)
        self._push_card(updated_card, commit_message)
        return f"https://huggingface.co/{self.repo_id}"

    # ------------------------------------------------------------------
    # Read compliance section
    # ------------------------------------------------------------------

    def read_compliance_section(self) -> Dict[str, Any]:
        """
        Read back the structured compliance metadata from the model card.

        Returns
        -------
        dict
            Parsed compliance metadata, or ``{}`` if no compliance section exists.
        """
        content = self._fetch_card_content()
        match = re.search(
            re.escape(_COMPLIANCE_JSON_START) + r"\s*(.*?)\s*" + re.escape(_COMPLIANCE_JSON_END),
            content,
            re.DOTALL,
        )
        if not match:
            return {}
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return {"raw": match.group(1).strip()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_card_content(self) -> str:
        try:
            path = self._hf.hf_hub_download(
                repo_id   = self.repo_id,
                filename  = "README.md",
                repo_type = self.repo_type,
                token     = self.token,
            )
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            # Model card doesn't exist yet — start blank
            return ""

    def _build_section(
        self,
        result: Dict[str, Any],
        annex_iv_url: Optional[str],
        auditor: Optional[str],
    ) -> str:
        faith   = result.get("faithfulness", {})
        suff    = faith.get("sufficiency", 0.0)
        comp    = faith.get("comprehensiveness", 0.0)
        f1      = faith.get("f1", 0.0)
        cat     = faith.get("category", "unknown")
        n_heads = result.get("n_heads", 0)

        if suff >= 0.90: badge = "✅ Excellent"
        elif suff >= 0.75: badge = "✅ Good"
        elif suff >= 0.50: badge = "⚠️ Marginal"
        else: badge = "❌ Poor"

        metadata: Dict[str, Any] = {
            "glassbox_version":     "4.2.6",
            "audit_timestamp_utc":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "auditor":              auditor or "automated",
            "model_repo_id":        self.repo_id,
            "faithfulness_sufficiency":    round(suff, 4),
            "faithfulness_comprehensiveness": round(comp, 4),
            "faithfulness_f1":      round(f1, 4),
            "faithfulness_category": cat,
            "circuit_n_heads":      n_heads,
            "compliance_status":    "COMPLIANT" if suff >= 0.75 else "NEEDS_REVIEW",
            "regulation":           "Regulation (EU) 2024/1689 — AI Act Annex IV",
            "annex_iv_report_url":  annex_iv_url or "",
        }

        url_line = f"\n- **Full Annex IV Report**: [{annex_iv_url}]({annex_iv_url})" if annex_iv_url else ""
        auditor_line = f"\n- **Auditor**: {auditor}" if auditor else ""

        section = textwrap.dedent(f"""
            {_COMPLIANCE_SECTION_HEADER}

            This model has been audited using [Glassbox AI](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) v4.2.6
            for mechanistic interpretability and EU AI Act Annex IV compliance.

            | Metric | Value |
            |---|---|
            | Causal Faithfulness (Sufficiency) | {suff:.1%} |
            | Comprehensiveness | {comp:.1%} |
            | Faithfulness F1 | {f1:.1%} |
            | Circuit Size | {n_heads} heads |
            | Behaviour Category | {cat.replace('_', ' ').title()} |
            | **Compliance Grade** | **{badge}** |

            **Regulation**: Regulation (EU) 2024/1689 — AI Act, Annex IV
            **Articles covered**: Art. 9 (Risk), Art. 11 (Documentation), Art. 13 (Transparency), Art. 15 (Accuracy), Art. 72 (Monitoring){url_line}{auditor_line}

            {_COMPLIANCE_JSON_START}
            {json.dumps(metadata, indent=2)}
            {_COMPLIANCE_JSON_END}
        """).strip()

        return "\n\n" + section + "\n"

    @staticmethod
    def _inject_section(card_content: str, new_section: str) -> str:
        """Replace existing compliance section or append to end."""
        # Remove existing section if present
        pattern = (
            re.escape(_COMPLIANCE_SECTION_HEADER)
            + r".*?"
            + re.escape(_COMPLIANCE_JSON_END)
            + r"[^\n]*\n?"
        )
        cleaned = re.sub(pattern, "", card_content, flags=re.DOTALL)
        return cleaned.rstrip() + new_section

    def _push_card(self, content: str, commit_message: str) -> None:
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            self._hf.upload_file(
                path_or_fileobj = tmp_path,
                path_in_repo    = "README.md",
                repo_id         = self.repo_id,
                repo_type       = self.repo_type,
                token           = self.token,
                commit_message  = commit_message,
            )
        finally:
            os.unlink(tmp_path)
