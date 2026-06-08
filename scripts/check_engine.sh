#!/usr/bin/env bash
#
# check_engine.sh — reproduce the CI engine-test environment locally and run it.
#
# What it does:
#   1. Creates a throwaway virtualenv (.ci_repro_venv).
#   2. Installs the EXACT verified-good dependency stack (the versions the engine
#      is known to pass on), then installs glassbox with --no-deps.
#   3. Prints the resolved versions and confirms transformer_lens is the REAL
#      package (has a __file__), not a MagicMock test stub.
#   4. Sanity-loads GPT-2 and asserts it is a real HookedTransformer — this is
#      the specific regression that broke CI (a stub model poisoning the suite).
#   5. Runs the full test suite with coverage (gate off — it just reports %).
#
# If this prints "RESULT: PASS", CI (same recipe) should be green too.
#
#   Usage:    bash scripts/check_engine.sh
#   Cleanup:  rm -rf .ci_repro_venv
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV=".ci_repro_venv"
PY="${PYTHON:-python3}"

echo "==> Creating throwaway venv: $VENV"
rm -rf "$VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null

echo "==> Installing pinned, verified-good stack"
pip install \
  "numpy==1.26.4" "scipy==1.17.1" "einops==0.8.2" \
  "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6" \
  pytest pytest-cov reportlab ipywidgets

echo "==> Installing glassbox (no deps)"
pip install -e . --no-deps >/dev/null

echo "==> Resolved versions"
python - <<'PY'
import torch, numpy, transformer_lens
print(f"  torch             {torch.__version__}")
print(f"  numpy             {numpy.__version__}")
print(f"  transformer_lens  {getattr(transformer_lens, '__file__', 'NONE — STUB!')}")
PY

echo "==> Sanity: a REAL HookedTransformer must load (not a MagicMock)"
python - <<'PY'
from transformer_lens import HookedTransformer
m = HookedTransformer.from_pretrained("gpt2")
mod = type(m).__module__
assert mod.startswith("transformer_lens"), f"model is {type(m)} from {mod} — stub poisoning!"
print(f"  OK — real model: {type(m).__name__} (n_layers={m.cfg.n_layers})")
PY

echo "==> Full test suite with coverage (gate off)"
set +e
pytest --cov=glassbox --cov-report=term-missing --cov-fail-under=0 -q
STATUS=$?
set -e

deactivate
echo
if [ "$STATUS" -eq 0 ]; then
  echo "RESULT: PASS — engine green on the pinned stack. CI should match."
else
  echo "RESULT: FAIL (exit $STATUS) — paste the summary line shown above."
fi
echo "(cleanup: rm -rf $VENV)"
exit "$STATUS"
