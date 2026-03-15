import argparse
import sys

BANNER = """
  +===================================+
  |   G L A S S B O X  2 . 0         |
  |  Mechanistic Interpretability CLI |
  +===================================+
"""


def _run_analyze(args):
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2

    print(BANNER)
    print(f"  Model  : {args.model}")
    print(f"  Prompt : {args.prompt!r}")
    print(f"  Correct: {args.correct!r}   Incorrect: {args.incorrect!r}\n")

    # BUG FIX: original code did GlassboxV2(args.model) — passing the string
    # "gpt2" directly.  GlassboxV2.__init__ immediately calls model.cfg.n_layers
    # which doesn't exist on a string -> AttributeError on every CLI run.
    # FIX: load the HookedTransformer first, then pass the object.
    model = HookedTransformer.from_pretrained(args.model)
    gb    = GlassboxV2(model)

    result  = gb.analyze(args.prompt, args.correct, args.incorrect)
    faith   = result["faithfulness"]

    # BUG FIX: original code used faith['suff'] and faith['comp'].
    # Those keys do not exist.  analyze() returns faith['sufficiency']
    # and faith['comprehensiveness'].  Every CLI run crashed with KeyError.
    print(f"  Sufficiency      : {faith['sufficiency']:.1%}")
    print(f"  Comprehensiveness: {faith['comprehensiveness']:.1%}")
    print(f"  F1-score         : {faith['f1']:.1%}")
    print(f"  Category         : {faith['category']}")

    # Surface the approximation flag so the user knows what sufficiency means
    if faith.get("suff_is_approx"):
        print(f"  Note             : Sufficiency is a first-order Taylor approximation.")
        print(f"                     See class docstring for details.\n")
    else:
        print()

    print(f"  {'Head':<12} {'Attribution':>12}")
    print(f"  {'-'*12} {'-'*12}")

    # BUG FIX: original code did circuit.items() — circuit is a List[Tuple],
    # not a Dict.  Calling .items() on a list raises AttributeError.
    # FIX: iterate the list of tuples, look up scores from attributions dict.
    attrs = result["attributions"]
    for (layer, head) in result["circuit"]:
        score = attrs.get(str((layer, head)), 0.0)
        print(f"  L{layer:02d}H{head:02d}      {score:>12.4f}")


def main():
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="Glassbox 2.0 — Mechanistic Interpretability",
        epilog="""Examples:
  glassbox analyze --prompt "When Mary and John went to the store, John gave a gift to" \\
                   --correct "Mary" --incorrect "John"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("analyze", help="Analyze a prompt circuit")
    p.add_argument("--prompt",    required=True, help="Input prompt")
    p.add_argument("--correct",   required=True, help="Correct completion token")
    p.add_argument("--incorrect", required=True, help="Incorrect completion token")
    p.add_argument("--model",     default="gpt2", help="Model name (default: gpt2)")

    args = parser.parse_args()

    if args.cmd == "analyze":
        _run_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
