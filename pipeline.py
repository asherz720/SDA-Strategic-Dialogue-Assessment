"""End-to-end SDA evaluation: validate -> score -> correlate with humans.

Each model annotation CSV is one witness/trial. Pass one or more, with the
matching human-annotation CSV(s), and get a results table combining the SDA
score summaries with agreement-against-humans. Choose per-witness rows or a
single mean across witnesses, and optionally write it to CSV.

    # one witness, print to screen
    python pipeline.py --model "model_annotations/WMT_D/gpt-4o-mini/JM_ofshe.csv" \
                       --human "human annotations/WMT_D_annotations.csv"

    # several witnesses -> per-witness rows, saved to CSV
    python pipeline.py --model model_annotations/enron_p/gpt-4o-mini/enron_prosecution_*.csv \
                       --human "human annotations/enron_prosecution_"*.csv \
                       --level witness --output enron_4omini.csv

    # also dump the per-turn scored tables
    python pipeline.py --model m1.csv m2.csv --human h1.csv h2.csv --scores-out scored/

``--human`` may be a single file (reused for every model) or one per model.
To benchmark a whole model by id across all cases at once, use ``benchmark.py``
instead. For programmatic use, see ``sda.metrics.score`` and
``sda.agreement.evaluate_dataset``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from sda import compare_human_llm, score, validate
from sda.agreement import REPORT_METRICS, mean_across


def witness_label(model_df, path):
    """Human-readable witness/trial name: the witness_name column, else filename."""
    if "witness_name" in model_df.columns and model_df["witness_name"].notna().any():
        return str(model_df["witness_name"].dropna().iloc[0])
    return Path(path).stem


def evaluate_files(model_paths, human_paths, *, level="mean", scores_out=None, verbose=False):
    """Evaluate one or more (model, human) file pairs into a results DataFrame."""
    model_paths = list(model_paths)
    human_paths = list(human_paths)
    if len(human_paths) == 1:
        human_paths = human_paths * len(model_paths)
    if len(human_paths) != len(model_paths):
        raise ValueError(
            f"Got {len(model_paths)} model file(s) but {len(human_paths)} human file(s); "
            "pass one human file per model, or a single shared human file."
        )
    if scores_out:
        os.makedirs(scores_out, exist_ok=True)

    rows = []
    for mp, hp in zip(model_paths, human_paths):
        model_df = pd.read_csv(mp)
        human_df = pd.read_csv(hp)

        report = validate(model_df, emit_warnings=False)
        if not report.ok:
            print(f"[warning] {mp} failed validation:")
            print("  " + report.summary().replace("\n", "\n  "))

        scored = score(model_df)
        if scores_out:
            scored.to_csv(os.path.join(scores_out, Path(mp).name), index=False)

        row = {"witness": witness_label(model_df, mp), "n_turns": len(scored)}
        agreement = compare_human_llm(scored, human_df, verbose=verbose)
        row.update({col: agreement[k] for k, col in REPORT_METRICS.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    if level == "mean":
        df = mean_across(df, "witness", sum_cols=["n_turns"])
    return df


def main():
    ap = argparse.ArgumentParser(description="SDA end-to-end evaluation.")
    ap.add_argument("--model", nargs="+", required=True, help="model annotation CSV(s), one per witness")
    ap.add_argument("--human", nargs="+", required=True, help="human annotation CSV(s): one per model, or one shared")
    ap.add_argument("--level", choices=["mean", "witness"], default="mean",
                    help="aggregate across witnesses (mean) or report each (witness)")
    ap.add_argument("--output", help="write the results table to this CSV")
    ap.add_argument("--scores-out", help="directory to also write each per-turn scored table into")
    args = ap.parse_args()

    df = evaluate_files(args.model, args.human, level=args.level, scores_out=args.scores_out)
    print(df.round(3).to_string(index=False))
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
