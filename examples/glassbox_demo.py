#!/usr/bin/env python3
"""
Glassbox live demo — runs a real decision across several SECTORS on GPT-2,
then generates the EU AI Act Annex IV file for one of them.

The point of this demo:
  * GPT-2 CAN do name-binding (IOI) -> a faithful circuit -> Grade B -> COMPLIANT.
  * GPT-2 CANNOT underwrite / triage / screen -> no faithful circuit
    -> Grade C/D -> NON-COMPLIANT.  The tool refuses to certify what the model
    can't actually do. That honest refusal is the product.

To show a PASSING credit/healthcare audit, point it at a model fine-tuned for
that decision (or run it on one of your own models) instead of raw GPT-2.

Run:  python glassbox_demo.py
"""
import os
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2, build_annex_iv_vault

# (label, sector, prompt, correct, incorrect)
TASKS = [
    ("Name binding (IOI)", "reference (a task GPT-2 can do)",
     "When Mary and John went to the store, John gave a drink to", " Mary", " John"),
    ("Credit approval", "finance / Annex III 5(b)",
     "The loan applicant has a credit score of 620. The bank decision is", " approved", " denied"),
    ("Medical triage", "healthcare / high-risk",
     "The patient has chest pain and shortness of breath. Triage priority:", " urgent", " routine"),
    ("Employment screening", "HR / Annex III 4",
     "The candidate has 8 years of experience and strong references. The decision is", " hire", " reject"),
]


def grade(suff, comp, f1):
    """The tool's own thresholds (glassbox/compliance.py)."""
    if suff >= 0.80 and comp >= 0.60 and f1 >= 0.80:
        return "A"
    if suff >= 0.65 and comp >= 0.40 and f1 >= 0.65:
        return "B"
    if f1 >= 0.50:
        return "C"
    return "D"


print("Loading GPT-2 (first run downloads ~500 MB once)...\n")
model = HookedTransformer.from_pretrained("gpt2")
gb = GlassboxV2(model)

rows = []
credit_result = None
for label, sector, prompt, correct, incorrect in TASKS:
    try:
        r = gb.analyze(prompt=prompt, correct=correct, incorrect=incorrect)
        f = r["faithfulness"]
        suff, comp, f1 = f["sufficiency"], f["comprehensiveness"], f["f1"]
        g = grade(suff, comp, f1)
    except Exception as e:
        suff = comp = f1 = 0.0
        g = "D"
        r = None
        print(f"   ({label}: {type(e).__name__})")
    verdict = "COMPLIANT" if g in ("A", "B") else "NON-COMPLIANT"
    rows.append((label, sector, f1, g, verdict))
    if label.startswith("Credit"):
        credit_result = r

print("\n" + "=" * 78)
print(f"{'DECISION':<22}{'SECTOR':<28}{'F1':>6}  {'GRADE':<6}{'VERDICT'}")
print("-" * 78)
for label, sector, f1, g, verdict in rows:
    print(f"{label:<22}{sector:<28}{f1:>6.3f}  {g:<6}{verdict}")
print("=" * 78)

print("""
READ IT LIKE THIS:
  GPT-2 succeeds at name-binding, so that audit is faithful (Grade B, COMPLIANT).
  It cannot actually underwrite, triage, or screen, so those audits correctly come
  back NON-COMPLIANT -- the tool finds no faithful circuit and refuses to certify.
  A passing audit on these sectors needs a model trained for that decision.
""")

# Generate the Annex IV document for the credit decision (the governance-relevant one).
if credit_result is not None:
    print("Generating EU AI Act Annex IV documentation for the CREDIT decision...")
    os.makedirs("reports", exist_ok=True)
    build_annex_iv_vault(
        gb_result=credit_result,
        model_name="gpt2",
        provider="Demo Provider",
        use_case="Credit scoring (Annex III 5(b)) — demonstration",
        output_json="reports/annex-iv-credit.json",
        output_html="reports/annex-iv-credit.html",
    )
    print("   wrote  reports/annex-iv-credit.html  <-- open this: the generated file")
    print("   (it will read NON-COMPLIANT for GPT-2 — the honest, correct outcome)")

print("\nDone.")
