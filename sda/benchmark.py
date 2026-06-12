"""Benchmark a model across the whole CPD dataset in one call.

Relies on the normalized layout ``model_annotations/<case>/<model_id>/<witness>.csv``
and pairs each witness file to its human annotations automatically:

* ``WMT_D`` / ``WMT_P`` share one human file (``<case>_annotations.csv``);
* every other case pairs by filename (``enron_prosecution_1.csv`` ↔ same name).

Typical use::

    from sda.benchmark import evaluate_model, leaderboard
    evaluate_model("gpt-4o-mini", level="case")     # per-case rows + overall mean
    leaderboard(level="overall")                     # one row per model
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from .agreement import REPORT_METRICS, compare_human_llm, mean_across
from .metrics import score
from .validate import validate

MODEL_ROOT = "model_annotations"
HUMAN_DIR = "human annotations"
CASES = ["WMT_D", "WMT_P", "enron_d", "enron_p", "simpson_d", "simpson_p"]


def human_file_for(case, witness_filename):
    """Locate the human annotation file for a (case, witness) pair."""
    if case in ("WMT_D", "WMT_P"):
        return f"{HUMAN_DIR}/{case}_annotations.csv"
    return f"{HUMAN_DIR}/{witness_filename}"


def list_models():
    """All model ids present (a model id is a directory under any case)."""
    ids = {Path(d).name for d in glob.glob(f"{MODEL_ROOT}/*/*/")}
    return sorted(ids)


def discover(model_id):
    """Yield (case, model_path, human_path) for every witness of ``model_id``."""
    for mp in sorted(glob.glob(f"{MODEL_ROOT}/*/{model_id}/*.csv")):
        p = Path(mp)
        case = p.parts[1]
        yield case, mp, human_file_for(case, p.name)


def evaluate_pair(model_df, human_df, *, witness=None, case=None, verbose=False):
    """One results row: avg correlation/agreement with humans on each metric."""
    scored = score(model_df)
    row = {"case": case, "witness": witness, "n_turns": len(scored)}
    agreement = compare_human_llm(scored, human_df, verbose=verbose)
    row.update({col: agreement[k] for k, col in REPORT_METRICS.items()})
    return row


def evaluate_model(model_id, *, level="overall", verbose=False, warn=True):
    """Evaluate one model across all its witnesses.

    ``level``: ``"witness"`` (one row per witness), ``"case"`` (one row per case
    plus an overall row), or ``"overall"`` (a single mean row).
    """
    rows = []
    for case, mp, hp in discover(model_id):
        model_df = pd.read_csv(mp)
        if warn:
            rep = validate(model_df, emit_warnings=False)
            if not rep.ok:
                print(f"[warning] {mp}:\n  " + rep.summary().replace("\n", "\n  "))
        rows.append(evaluate_pair(model_df, pd.read_csv(hp),
                                  witness=Path(mp).stem, case=case, verbose=verbose))
    if not rows:
        raise ValueError(f"No files found for model id '{model_id}' under {MODEL_ROOT}/*/")
    df = pd.DataFrame(rows)

    if level == "witness":
        return df
    if level == "overall":
        out = mean_across(df, "witness", sum_cols=["n_turns"])
        out.insert(0, "model", model_id)
        return out.drop(columns=["witness"])
    if level == "case":
        per_case = [mean_across(g, "witness", sum_cols=["n_turns"]).assign(case=c)
                    for c, g in df.groupby("case")]
        overall = mean_across(df, "witness", sum_cols=["n_turns"]).assign(case="ALL")
        out = pd.concat(per_case + [overall], ignore_index=True)
        cols = ["case"] + [c for c in out.columns if c not in ("case", "witness")]
        return out[cols]
    raise ValueError("level must be 'witness', 'case', or 'overall'")


def leaderboard(*, level="overall", models=None, verbose=False):
    """One row per model (``level='overall'``) — the cross-model comparison table.

    With ``level='case'`` returns one row per (model, case).
    """
    models = models or list_models()
    out = []
    for mid in models:
        df = evaluate_model(mid, level=("overall" if level == "overall" else "case"),
                            verbose=verbose, warn=False)
        if level == "overall":
            out.append(df)
        else:
            out.append(df.assign(model=mid))
    res = pd.concat(out, ignore_index=True)
    if level != "overall":
        res = res[["model"] + [c for c in res.columns if c != "model"]]
    return res
