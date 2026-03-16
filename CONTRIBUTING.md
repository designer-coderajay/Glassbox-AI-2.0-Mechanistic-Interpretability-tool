# Contributing to Glassbox

Thank you for contributing. Glassbox is research software — correctness and mathematical
integrity matter more than feature count. A few principles:

## Before you open a PR

1. **Every new method needs a citation** if it implements or adapts a published technique.
   Add it to the module docstring and the references block at the top of `core.py`.
2. **Every approximation must be disclosed.** If your method produces an approximation,
   add an `APPROXIMATION NOTE (disclosed)` block in the docstring explaining exactly
   what is approximated and when it degrades.
3. **Write tests first.** Add your tests to `tests/test_engine.py` before implementing.
   Use the existing test class structure as a template.
4. **Update the complexity table** in the `core.py` module docstring with the pass count
   for your new method.

## Development setup

```bash
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
pip install -e ".[dev]"
```

## Running tests

```bash
# Fast tests only (skips @pytest.mark.slow)
pytest tests/ -m "not slow" -v

# All tests (loads GPT-2 multiple times — takes ~5 min on CPU)
pytest tests/ -v
```

## Code style

- Line length: 100 characters.
- Type annotations: preferred but not required for internal helpers.
- Variable names: use maths-style names (`W_Q`, `d_head`, `n_layers`) consistent with
  Elhage et al. (2021) notation throughout the codebase.

## Adding a new analysis method

1. Add it as a method on `GlassboxV2` in `core.py`.
2. Update the `GlassboxV2` class docstring (Public API section).
3. Update the complexity table in the module docstring.
4. Export it from `__init__.py` if it is a standalone class.
5. Add tests in `tests/test_engine.py`.
6. Update `README.md` — add a row to the "What's Novel" table and a code example.
7. Add a CHANGELOG entry under the version it ships in.

## Bug reports

Please include:
- The exact prompt, correct token, and distractor token you used.
- The model name.
- The full traceback.
- The Glassbox version (`python -c "import glassbox; print(glassbox.__version__)"`).

## Mathematical questions

Open an issue with the `math` label. Include the relevant formula and what you believe
the discrepancy is. We take mathematical accuracy seriously and will respond promptly.
