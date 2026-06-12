"""Benchmark a model (or all models) on the CPD dataset.

    python benchmark.py --list                       # show available model ids
    python benchmark.py --model gpt-4o-mini           # one model, overall score
    python benchmark.py --model qwq-32b --level case  # per-case + overall rows
    python benchmark.py --all --output leaderboard.csv  # one row per model

Model ids and the normalized layout come from model_annotations/<case>/<model_id>/.
Human files are paired automatically (see sda.benchmark).
"""

from __future__ import annotations

import argparse

from sda.benchmark import evaluate_model, leaderboard, list_models


def main():
    ap = argparse.ArgumentParser(description="SDA benchmark runner.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="model id to evaluate (e.g. gpt-4o-mini)")
    g.add_argument("--all", action="store_true", help="evaluate every model -> leaderboard")
    g.add_argument("--list", action="store_true", help="list available model ids and exit")
    ap.add_argument("--level", default=None,
                    help="single model: witness|case|overall (default overall); --all: overall|case (default overall)")
    ap.add_argument("--output", help="write the table to this CSV")
    args = ap.parse_args()

    if args.list:
        print("\n".join(list_models()))
        return

    if args.all:
        df = leaderboard(level=args.level or "overall")
    else:
        df = evaluate_model(args.model, level=args.level or "overall")

    print(df.round(3).to_string(index=False))
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
