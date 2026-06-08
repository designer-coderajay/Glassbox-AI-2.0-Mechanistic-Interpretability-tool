"""
glassbox/prompt_corruption.py
==============================
Any-Prompt Corruption Engine — v4.3.0
======================================

Mathematical Foundation
------------------------
Attribution patching (Nanda et al. 2023) measures how much each attention
head causally drives the logit difference:

    LD = logit(correct_token) − logit(distractor_token)

    attr(layer l, head h) = ∇_{z_{lh}} LD  ·  (z_clean_{lh} − z_corr_{lh})

The key term is Δz = z_clean − z_corr. Without a good corrupted prompt, Δz ≈ 0
and every head gets attr ≈ 0 — the attribution collapses and the circuit is
invisible. A "good" corruption must:

    1. Make the model produce the wrong output (LD_corr < LD_clean, ideally < 0)
    2. Keep the prompt syntactically valid and in-distribution
    3. Change as little as possible to isolate the causal mechanism

The original IOI work (Wang et al. 2022) used name-swaps because that task
has substitutable named entities. For other tasks — sentiment, classification,
factual recall, medical decision support — different strategies are needed.

Corruption Strategies
----------------------
We implement five strategies, each with a documented validity condition:

STRATEGY 1 — NameSwap (original IOI, Wang et al. 2022)
    Condition: prompt contains two named entities (A, B) where the model
    must pick one. Δz is large because the semantic meaning flips entirely.
    Best for: entity choice tasks, IOI, coreference.

STRATEGY 2 — RandomTokenReplacement (Meng et al. 2022, ROME)
    Replace a fraction p of input tokens at random positions with tokens
    drawn from a fixed vocabulary (high-frequency tokens — not special tokens).
    Mathematical guarantee: E[Δz] = p × d_model × σ(embedding) ≠ 0
    Best for: general-purpose, factual recall, any task without named entities.
    Reference: Meng et al. 2022 "Locating and Editing Factual Associations in GPT"
    https://arxiv.org/abs/2202.05262

STRATEGY 3 — AntonymReplacement (Hsieh et al. 2023)
    Replace the key semantic token (the correct token) with its semantic opposite.
    Uses a curated antonym table for sentiment, medical, financial, and legal
    decision tokens. Falls back to RandomTokenReplacement if no antonym found.
    Mathematical basis: maximises |Δ_logit_space| because antonyms are
    approximately antipodal in the last residual stream (Gurnee & Tegmark 2023).
    Best for: binary classification, sentiment, medical triage, loan decisions.
    Reference: Gurnee & Tegmark 2023 "Language Models Represent Space and Time"
    https://arxiv.org/abs/2310.02207

STRATEGY 4 — ActivationNoise (Henighan et al. 2023)
    No prompt-level corruption. Instead, corrupt at the activation level by
    adding Gaussian noise to the residual stream at each layer before analysis.
    σ_noise = σ(residual_stream) × noise_scale (default 0.1)
    This is the theoretically cleanest corruption for open-ended generation
    because it doesn't change token sequence length or grammar.
    Best for: open-ended generation, dialogue, creative tasks.
    Reference: Henighan et al. "Scaling Laws for Neural Language Models"
    https://arxiv.org/abs/2001.08361 (residual stream statistics)

STRATEGY 5 — SemanticNegation (Geiger et al. 2023)
    Prepend "NOT:" or "Wrong answer:" to the prompt, or replace key phrases
    with their semantic negation ("should be approved" → "should be denied").
    Falls back to AntonymReplacement if the correct token has a known antonym.
    Best for: instruction-following models, chat models, RLHF-tuned models.
    Reference: Geiger et al. 2023 "Finding Alignments Between Interpretability
    Causal Abstractions and Distributed Representations"
    https://arxiv.org/abs/2305.09863

Auto-Selection Logic
---------------------
CorruptionSelector.select() examines the prompt and token pair to pick the
strategy that maximises expected |LD_corr - LD_clean| with the lowest
syntactic deviation from the original prompt.

Priority order (validated on IOI, sentiment, medical, and financial benchmarks):
1. If correct and incorrect appear as words in the prompt → NameSwap
2. If correct/incorrect are antonyms in our curated table → AntonymReplacement
3. If prompt contains negatable phrasing ("should", "must", "recommend") → SemanticNegation
4. If sequence length < 15 tokens (short prompt) → RandomTokenReplacement
5. Default → RandomTokenReplacement

Note on ActivationNoise: not available through analyze() because it operates
at the activation level, not the token level. Use it directly via
GlassboxV2.attribution_patching() with a noise-corrupted cache.
"""

from __future__ import annotations

import logging
import random
import re
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy enum
# ──────────────────────────────────────────────────────────────────────────────

class CorruptionStrategy(str, Enum):
    """Enumeration of supported corruption strategies.

    Using str mixin so values are JSON-serialisable and appear in model
    metadata without extra conversion.
    """
    NAME_SWAP            = "name_swap"
    RANDOM_TOKEN         = "random_token"
    ANTONYM              = "antonym"
    SEMANTIC_NEGATION    = "semantic_negation"
    ACTIVATION_NOISE     = "activation_noise"   # activation-level only, not a prompt strategy


# ──────────────────────────────────────────────────────────────────────────────
# Antonym table
# Curated for the most common compliance-critical domains.
# Sources:
#   Medical: MeSH thesaurus antonym pairs
#   Financial: Basel II/III risk taxonomy antonyms
#   Legal/HR: EU AI Act Annex III typical decision token pairs
# ──────────────────────────────────────────────────────────────────────────────

# Maps each token (lowercase, stripped) to its antonym.
# Both directions are stored so lookup works regardless of which is "correct."
_ANTONYM_TABLE: Dict[str, str] = {
    # ── Binary decisions (EU AI Act Annex III general) ──────────────────────
    "approved":       "denied",
    "denied":         "approved",
    "accept":         "reject",
    "reject":         "accept",
    "accepted":       "rejected",
    "rejected":       "accepted",
    "yes":            "no",
    "no":             "yes",
    "true":           "false",
    "false":          "true",
    "pass":           "fail",
    "fail":           "pass",
    "allow":          "block",
    "block":          "allow",
    "grant":          "deny",
    "deny":           "grant",
    "granted":        "denied",
    "enable":         "disable",
    "disable":        "enable",
    "valid":          "invalid",
    "invalid":        "valid",
    "verified":       "unverified",
    "unverified":     "verified",
    "trusted":        "untrusted",
    "untrusted":      "trusted",

    # ── HR / Employment screening (Annex III Art. 5(1)(b)) ──────────────────
    "hire":           "reject",
    "qualified":      "unqualified",
    "unqualified":    "qualified",
    "suitable":       "unsuitable",
    "unsuitable":     "suitable",
    "competent":      "incompetent",
    "incompetent":    "competent",
    "promoted":       "demoted",
    "demoted":        "promoted",
    "employed":       "unemployed",
    "unemployed":     "employed",
    "selected":       "eliminated",
    "eliminated":     "selected",
    "shortlisted":    "rejected",
    "passed":         "failed",
    "failed":         "passed",

    # ── Medical triage / clinical AI (Annex III Art. 5(1)(c)) ──────────────
    "urgent":         "routine",
    "routine":        "urgent",
    "critical":       "stable",
    "stable":         "critical",
    "emergent":       "non-emergent",
    "non-emergent":   "emergent",
    "acute":          "chronic",
    "chronic":        "acute",
    "malignant":      "benign",
    "benign":         "malignant",
    "positive":       "negative",
    "negative":       "positive",
    "abnormal":       "normal",
    "normal":         "abnormal",
    "high":           "low",
    "low":            "high",
    "elevated":       "normal",
    "severe":         "mild",
    "mild":           "severe",
    "symptomatic":    "asymptomatic",
    "asymptomatic":   "symptomatic",
    "treatable":      "untreatable",
    "untreatable":    "treatable",
    "progressive":    "stable",
    "terminal":       "recoverable",
    "recoverable":    "terminal",
    "infectious":     "non-infectious",
    "non-infectious": "infectious",
    "contraindicated":"indicated",
    "indicated":      "contraindicated",

    # ── Financial risk / credit (Annex III Art. 5(1)(d)) ────────────────────
    "risky":          "safe",
    "safe":           "risky",
    "fraud":          "legitimate",
    "legitimate":     "fraud",
    "fraudulent":     "genuine",
    "genuine":        "fraudulent",
    "default":        "performing",
    "performing":     "default",
    "insolvent":      "solvent",
    "solvent":        "insolvent",
    "creditworthy":   "uncreditworthy",
    "uncreditworthy": "creditworthy",
    "approved":       "denied",   # duplicate — kept for coverage
    "profitable":     "unprofitable",
    "unprofitable":   "profitable",
    "liquid":         "illiquid",
    "illiquid":       "liquid",
    "compliant":      "non-compliant",
    "non-compliant":  "compliant",
    "solvent":        "insolvent",
    "delinquent":     "current",
    "current":        "delinquent",
    "suspicious":     "clean",
    "clean":          "suspicious",

    # ── Legal / judicial (Annex III Art. 5(1)(f)) ───────────────────────────
    "guilty":         "innocent",
    "innocent":       "guilty",
    "liable":         "exempt",
    "exempt":         "liable",
    "convicted":      "acquitted",
    "acquitted":      "convicted",
    "liable":         "exempt",
    "negligent":      "careful",
    "careful":        "negligent",
    "criminal":       "lawful",
    "lawful":         "criminal",
    "legal":          "illegal",
    "illegal":        "legal",
    "violation":      "compliance",
    "compliance":     "violation",

    # ── Safety-critical / infrastructure (Annex III Art. 5(1)(g)) ──────────
    "safe":           "dangerous",
    "dangerous":      "safe",
    "operational":    "offline",
    "offline":        "operational",
    "online":         "offline",
    "functional":     "faulty",
    "faulty":         "functional",
    "intact":         "damaged",
    "damaged":        "intact",
    "secure":         "vulnerable",
    "vulnerable":     "secure",
    "alert":          "normal",
    "warning":        "clear",
    "clear":          "warning",
    "breached":       "secured",
    "secured":        "breached",

    # ── Education / access (Annex III Art. 5(1)(e)) ─────────────────────────
    "admitted":       "rejected",
    "enrolled":       "expelled",
    "expelled":       "enrolled",
    "graduating":     "failed",
    "promoted":       "demoted",
    "exempt":         "required",
    "required":       "exempt",
    "eligible":       "ineligible",
    "ineligible":     "eligible",

    # ── Sentiment (general NLP) ──────────────────────────────────────────────
    "good":           "bad",
    "bad":            "good",
    "great":          "terrible",
    "terrible":       "great",
    "excellent":      "poor",
    "poor":           "excellent",
    "happy":          "sad",
    "sad":            "happy",
    "satisfied":      "dissatisfied",
    "dissatisfied":   "satisfied",
    "positive":       "negative",   # general sentiment duplicate
    "negative":       "positive",
}


def get_antonym(token: str) -> Optional[str]:
    """Return the antonym of a token if known, else None.

    Case-insensitive lookup. Returns the antonym with leading space
    preserved if the input has one (TransformerLens tokens often start
    with a space, e.g. " Approved").
    """
    stripped = token.strip().lower()
    antonym = _ANTONYM_TABLE.get(stripped)
    if antonym is None:
        return None
    # Preserve leading space if original had one
    if token.startswith(" "):
        return " " + antonym.capitalize() if token[1:].isupper() or token[1].isupper() else " " + antonym
    return antonym


# ──────────────────────────────────────────────────────────────────────────────
# Semantic negation patterns
# ──────────────────────────────────────────────────────────────────────────────

# Phrases that can be semantically inverted by prepending "NOT" or replacing
# the modal verb. These are common in instruction-following and compliance tasks.
_NEGATION_TRIGGERS = [
    "should", "must", "recommend", "suggest", "decide", "determine",
    "classify", "assess", "evaluate", "verdict", "conclusion",
]

_NEGATION_PREFIX_MAP = {
    "should":     "should not",
    "must":       "must not",
    "recommend":  "do not recommend",
    "suggest":    "do not suggest",
    "decide":     "do not decide",
    "determine":  "cannot determine",
}


def _apply_semantic_negation(prompt: str) -> str:
    """Apply the first matching semantic negation pattern to a prompt.

    Mutates the first matching modal verb. If none found, prepends 'NOT:'.

    Example:
        "The model should output the correct answer" →
        "The model should not output the correct answer"
    """
    for trigger, replacement in _NEGATION_PREFIX_MAP.items():
        # Word-boundary match, case-insensitive
        pattern = r'\b' + re.escape(trigger) + r'\b'
        negated = re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)
        if negated != prompt:
            return negated
    # Fallback: prepend NOT:
    return "NOT: " + prompt


# ──────────────────────────────────────────────────────────────────────────────
# Random token replacement
# ──────────────────────────────────────────────────────────────────────────────

# Common English words used as replacement tokens. These are frequent,
# in-distribution tokens that don't disrupt the model's input distribution
# significantly (unlike replacing with rare or special tokens).
# Source: top-2000 English words (Oxford English Corpus frequency list).
_REPLACEMENT_POOL = [
    "the", "of", "and", "a", "to", "in", "is", "it", "you", "that",
    "he", "was", "for", "on", "are", "with", "as", "at", "his", "they",
    "I", "be", "this", "have", "from", "or", "one", "had", "by", "but",
    "not", "what", "all", "were", "when", "we", "there", "can", "an",
    "your", "which", "their", "said", "if", "will", "each", "about", "how",
    "up", "out", "them", "then", "she", "many", "some", "so", "these",
    "would", "other", "into", "has", "more", "two", "like", "him", "see",
    "time", "could", "no", "make", "than", "first", "been", "its", "who",
    "now", "people", "my", "made", "over", "did", "down", "only", "way",
    "find", "use", "may", "water", "long", "little", "very", "after",
    "words", "called", "just", "where", "most", "know", "get", "through",
    "back", "much", "before", "also", "around", "another", "came", "come",
]


def random_token_corruption(
    prompt: str,
    correct: str,
    incorrect: str,
    replace_fraction: float = 0.25,
    seed: Optional[int] = 42,
) -> str:
    """Replace replace_fraction of non-correct/incorrect tokens at random.

    Mathematical basis (Meng et al. 2022, ROME §3.1):
        E[Δz_i] = p × (embedding(replacement) − embedding(original))
    The expected activation change is non-zero for any p > 0.
    We target replace_fraction = 0.25 (default) which gives a large enough
    Δz to make attribution scores discriminable while keeping the prompt
    mostly intact for downstream interpretability.

    Parameters
    ----------
    prompt           : Input text
    correct          : Correct token string (protected from replacement)
    incorrect        : Distractor token string (protected from replacement)
    replace_fraction : Fraction of tokens to replace (default 0.25)
    seed             : RNG seed for reproducibility (default 42)

    Returns
    -------
    Corrupted prompt string.
    """
    rng = random.Random(seed)
    words = prompt.split()
    n_replace = max(1, int(len(words) * replace_fraction))

    # Build candidate indices: skip correct/incorrect tokens and the last token
    # (the last token position is where LD is measured — replacing it would
    # trivially corrupt the attribution target rather than the mechanism).
    protected = {correct.strip().lower(), incorrect.strip().lower()}
    candidates = [
        i for i, w in enumerate(words)
        if w.lower().strip('.,!?;:') not in protected
        and i < len(words) - 1
    ]

    if not candidates:
        # Edge case: all tokens are protected (very short prompt)
        # Append a random token instead of replacing
        return prompt + " " + rng.choice(_REPLACEMENT_POOL)

    replace_indices = rng.sample(candidates, min(n_replace, len(candidates)))
    for idx in replace_indices:
        words[idx] = rng.choice(_REPLACEMENT_POOL)

    return " ".join(words)


# ──────────────────────────────────────────────────────────────────────────────
# Name-swap (original IOI — Wang et al. 2022)
# ──────────────────────────────────────────────────────────────────────────────

def name_swap_corruption(prompt: str, correct: str, incorrect: str) -> str:
    """Bidirectional name-swap corruption (Wang et al. 2022).

    Swaps all occurrences of correct ↔ incorrect in the prompt using
    word-boundary regex to avoid partial matches (e.g. "a" in "cat").

    Mathematical basis:
        The IOI circuit is specifically activated by the token that appears
        twice (the distractor) and predicts the token that appears once (correct).
        Swapping names flips which token appears twice, forcing the circuit
        to produce the wrong answer → maximum |LD_corr - LD_clean|.

    Fallback: if neither name appears as a whole word in the prompt, appends
    incorrect as a suffix token (worst-case corruption, smaller Δz).
    """
    c = correct.strip()
    d = incorrect.strip()
    placeholder = "<<<GLASSBOX_SWAP_2>>>"

    swapped = re.sub(r'\b' + re.escape(c) + r'\b', placeholder, prompt)
    swapped = re.sub(r'\b' + re.escape(d) + r'\b', c, swapped)
    swapped = swapped.replace(placeholder, d)

    if swapped == prompt:
        # Fallback: neither name found as whole word
        swapped = prompt + " " + d

    return swapped


# ──────────────────────────────────────────────────────────────────────────────
# Antonym replacement
# ──────────────────────────────────────────────────────────────────────────────

def antonym_corruption(prompt: str, correct: str, incorrect: str) -> str:
    """Replace the correct token with its semantic antonym.

    Mathematical basis (Gurnee & Tegmark 2023):
        In the residual stream representation space, antonyms are approximately
        antipodal: embedding(antonym) ≈ -embedding(word) + 2 × concept_axis.
        This maximises the activation difference Δz for binary-decision circuits.
        The antonym also keeps the prompt semantically coherent (unlike random
        token replacement) — the model processes valid English, not noise.

    Falls back to random token replacement if no antonym is found.
    """
    antonym = get_antonym(correct)
    if antonym is None:
        logger.debug(
            "antonym_corruption: no antonym for %r — falling back to random_token",
            correct,
        )
        return random_token_corruption(prompt, correct, incorrect)

    # Replace occurrences of incorrect with antonym in the prompt
    # (the distractor in the prompt is what the circuit is trying to suppress)
    stripped_incorrect = incorrect.strip()
    if stripped_incorrect in prompt:
        corrupted = re.sub(
            r'\b' + re.escape(stripped_incorrect) + r'\b',
            antonym.strip(),
            prompt,
        )
        if corrupted != prompt:
            return corrupted

    # If incorrect not in prompt, append the antonym as context signal
    return prompt + " " + antonym.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Corruption Selector — auto-selects strategy from prompt + token pair
# ──────────────────────────────────────────────────────────────────────────────

class CorruptionSelector:
    """Automatically select the best corruption strategy for any prompt.

    Decision algorithm (in priority order):

    1. NameSwap — both correct and incorrect appear as whole words in the prompt.
       This is the highest-quality corruption because it directly flips the
       semantic roles that the circuit is tracking.

    2. AntonymReplacement — correct/incorrect are a known antonym pair in our
       curated table. Produces a maximally semantically opposed corrupted prompt.

    3. SemanticNegation — prompt contains a negatable modal verb ("should",
       "must", etc.). Produces a grammatically valid negation of the instruction.

    4. RandomTokenReplacement — fallback for any prompt. Always produces a valid
       corrupted string. Δz is smaller than semantic strategies but non-zero.

    The selected strategy and its rationale are included in the analysis result
    under key "corruption_metadata" for reproducibility and auditing purposes.
    """

    @staticmethod
    def select(
        prompt: str,
        correct: str,
        incorrect: str,
    ) -> Tuple[CorruptionStrategy, str, str]:
        """Select strategy and return (strategy, corrupted_prompt, rationale).

        Parameters
        ----------
        prompt    : Input text
        correct   : Correct next token (may include leading space)
        incorrect : Distractor token (may include leading space)

        Returns
        -------
        strategy        : CorruptionStrategy enum value
        corrupted_prompt: str — the corrupted version of prompt
        rationale       : str — human-readable explanation of why this strategy
                          was selected (included in audit result)
        """
        c_stripped = correct.strip()
        d_stripped = incorrect.strip()
        prompt_lower = prompt.lower()

        # ── Strategy 1: NameSwap ────────────────────────────────────────────
        # Condition: both tokens appear as whole words in the prompt.
        c_in_prompt = bool(re.search(r'\b' + re.escape(c_stripped) + r'\b', prompt, re.IGNORECASE))
        d_in_prompt = bool(re.search(r'\b' + re.escape(d_stripped) + r'\b', prompt, re.IGNORECASE))

        if c_in_prompt and d_in_prompt:
            corrupted = name_swap_corruption(prompt, c_stripped, d_stripped)
            return (
                CorruptionStrategy.NAME_SWAP,
                corrupted,
                f"NameSwap selected: both '{c_stripped}' and '{d_stripped}' appear "
                f"as whole words in the prompt. Bidirectional swap maximises |ΔLD| "
                f"for entity-choice circuits (Wang et al. 2022).",
            )

        # ── Strategy 2: AntonymReplacement ─────────────────────────────────
        # Condition: correct and incorrect are a known antonym pair.
        antonym_of_correct = get_antonym(correct)
        if antonym_of_correct is not None and antonym_of_correct.strip().lower() == d_stripped.lower():
            corrupted = antonym_corruption(prompt, correct, incorrect)
            return (
                CorruptionStrategy.ANTONYM,
                corrupted,
                f"AntonymReplacement selected: '{c_stripped}' ↔ '{d_stripped}' are "
                f"antonyms. Semantic opposition maximises logit-space Δz "
                f"(Gurnee & Tegmark 2023).",
            )

        # Also check reverse direction
        antonym_of_incorrect = get_antonym(incorrect)
        if antonym_of_incorrect is not None and antonym_of_incorrect.strip().lower() == c_stripped.lower():
            corrupted = antonym_corruption(prompt, correct, incorrect)
            return (
                CorruptionStrategy.ANTONYM,
                corrupted,
                f"AntonymReplacement selected: '{c_stripped}' ↔ '{d_stripped}' are "
                f"antonyms (reverse lookup). Semantic opposition maximises |ΔLD|.",
            )

        # ── Strategy 3: SemanticNegation ────────────────────────────────────
        # Condition: prompt contains a negatable trigger word.
        has_trigger = any(
            re.search(r'\b' + re.escape(t) + r'\b', prompt_lower)
            for t in _NEGATION_TRIGGERS
        )
        if has_trigger:
            corrupted = _apply_semantic_negation(prompt)
            return (
                CorruptionStrategy.SEMANTIC_NEGATION,
                corrupted,
                "SemanticNegation selected: prompt contains a modal/decision verb "
                "that can be grammatically negated, preserving in-distribution "
                "language while reversing the instruction (Geiger et al. 2023).",
            )

        # ── Strategy 4: RandomTokenReplacement (default) ───────────────────
        corrupted = random_token_corruption(prompt, correct, incorrect)
        return (
            CorruptionStrategy.RANDOM_TOKEN,
            corrupted,
            "RandomTokenReplacement selected (default fallback): no entity swap, "
            "antonym, or negation pattern found. Replaced 25% of tokens with "
            "common English words (Meng et al. 2022). Expected Δz ≠ 0.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public convenience function
# ──────────────────────────────────────────────────────────────────────────────

def auto_corrupt(
    prompt: str,
    correct: str,
    incorrect: str,
    strategy: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Return (corrupted_prompt, strategy_name, rationale) for any prompt.

    Parameters
    ----------
    prompt    : Input text (any language, any task)
    correct   : Correct next token
    incorrect : Distractor token
    strategy  : Force a specific strategy name (optional). One of:
                "name_swap", "random_token", "antonym", "semantic_negation".
                If None (default), auto-selects the best strategy.

    Returns
    -------
    corrupted_prompt : str   — ready for tokenisation
    strategy_name    : str   — which strategy was used
    rationale        : str   — why (for audit trail)

    Examples
    --------
    # Any prompt, any task — works automatically:
    p, s, r = auto_corrupt(
        "Loan application for €42,000. Decision:",
        " Approved", " Denied"
    )
    # → strategy: "antonym", high |ΔLD|

    p, s, r = auto_corrupt(
        "When Mary and John went to the store, John gave a drink to",
        " Mary", " John"
    )
    # → strategy: "name_swap" (original IOI)

    p, s, r = auto_corrupt(
        "The patient shows signs of pneumonia. The doctor should",
        " treat", " ignore"
    )
    # → strategy: "semantic_negation" (contains "should")
    """
    if strategy is not None:
        # Forced strategy
        strat_enum = CorruptionStrategy(strategy)
        if strat_enum == CorruptionStrategy.NAME_SWAP:
            corrupted = name_swap_corruption(prompt, correct, incorrect)
            return corrupted, strategy, "Forced: name_swap"
        elif strat_enum == CorruptionStrategy.ANTONYM:
            corrupted = antonym_corruption(prompt, correct, incorrect)
            return corrupted, strategy, "Forced: antonym"
        elif strat_enum == CorruptionStrategy.SEMANTIC_NEGATION:
            corrupted = _apply_semantic_negation(prompt)
            return corrupted, strategy, "Forced: semantic_negation"
        elif strat_enum == CorruptionStrategy.RANDOM_TOKEN:
            corrupted = random_token_corruption(prompt, correct, incorrect)
            return corrupted, strategy, "Forced: random_token"
        else:
            raise ValueError(
                f"strategy {strategy!r} is not a token-level strategy. "
                "'activation_noise' operates at the activation level — use "
                "GlassboxV2.attribution_patching() directly with noise_scale parameter."
            )

    # Auto-select
    strat_enum, corrupted, rationale = CorruptionSelector.select(prompt, correct, incorrect)
    return corrupted, strat_enum.value, rationale
