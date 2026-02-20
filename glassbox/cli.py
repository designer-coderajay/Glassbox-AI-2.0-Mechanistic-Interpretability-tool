import argparse, sys, textwrap

BANNER = """
  ╔═══════════════════════════════════╗
  ║   ⬡  G L A S S B O X  2 . 0  ⬡  ║
  ║  Mechanistic Interpretability CLI ║
  ╚═══════════════════════════════════╝
"""

def _run_analyze(args):
    from glassbox import GlassboxV2
    print(BANNER)
    print(f"  Model  : {args.model}")
    print(f"  Prompt : {args.prompt!r}")
    print(f"  Correct: {args.correct!r}   Incorrect: {args.incorrect!r}\n")

    gb     = GlassboxV2(args.model)
    result = gb.analyze(args.prompt, args.correct, args.incorrect)
    circuit = result["circuit"]
    faith   = result["faithfulness"]

    print(f"  Sufficiency      : {faith['suff']:.1%}")
    print(f"  Comprehensiveness: {faith['comp']:.1%}")
    print(f"  F1-score         : {faith['f1']:.1%}")
    print(f"  Category         : {faith['category']}\n")

    print(f"  {'Head':<12} {'Attribution':>12}")
    print(f"  {'-'*12} {'-'*12}")
    for (layer, head), score in sorted(circuit.items(), key=lambda x: -x[1]):
        print(f"  L{layer:02d}H{head:02d}      {score:>12.4f}")


def main():
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="Glassbox 2.0 — Mechanistic Interpretability",
        epilog="""Examples:
  glassbox analyze --prompt "When Mary and John went to the store, John gave a gift to" \\
                   --correct " Mary" --incorrect " John"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("analyze", help="Analyze a prompt circuit")
    p.add_argument("--prompt",    required=True,          help="Input prompt")
    p.add_argument("--correct",   required=True,          help="Correct completion token")
    p.add_argument("--incorrect", required=True,          help="Incorrect completion token")
    p.add_argument("--model",     default="gpt2",         help="Model name (default: gpt2)")

    args = parser.parse_args()
    if args.cmd == "analyze":
        _run_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
