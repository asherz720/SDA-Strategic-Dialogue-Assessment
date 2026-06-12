"""Single source of truth for the SDA annotation schema.

A user evaluates their own model however they like, then parses the model's
JSON output into a table with the *core fields* below — one row per dialogue
turn. Everything downstream (BaT/PaT/NRBaT, correlation with humans) is computed
by this package from those fields.

Design note on the quality maxim: ``quality_rate`` (truthfulness) is recorded
and is compared against human annotators, but it deliberately does **not** enter
the BaT/PaT/NRBaT scores. Only commitment, manner and relevance shape the
metrics. (Earlier code carried a dead ``quality_rate >= 3`` branch from a
discontinued 1-4 quality encoding; it has been removed.)
"""

from __future__ import annotations

# --- Core fields the user must supply (parsed from their model's output) -----
# name -> set of valid values (None means "any value of the right kind")
CORE_FIELDS = {
    "question": None,                    # merge key against human annotations
    "Commitment_value": {1, 2, 3, 4},    # 1 detrimental, 2 beneficial, 3 neutral, 4 none
    "quality_rate": {0, 1},              # truthfulness: 1 truthful, 0 not
    "consistency_value": {0, 1},         # 1 inconsistent, 0 consistent
    "relevance_rate": {1, 2, 3, 4},      # 1 very relevant ... 4 irrelevant
    "manner_rate": {1, 2, 3, 4},         # 1 very clear ... 4 unclear
    "outcome_value": {"Questioner", "Witness"},
}

# Required to compute scores / correlate.
REQUIRED_FIELDS = list(CORE_FIELDS.keys())

# Nice to have but not required by scoring.
OPTIONAL_FIELDS = {
    "outcome_reason": {1, 2, 3},   # 1 logical, 2 credibility, 3 emotional
    "belief": None,                # free-text rationale
    "answer": None,                # the witness turn being rated
}

# Score columns produced by sda.metrics.score().
SCORE_COLUMNS = [
    "NRA",
    "bat",
    "pat",
    "bat_cumsum",
    "pat_cumsum",
    "Z_BaT",
    "Z_PaT",
    "ZNRBaT",
    "net_ZNRBaT",
]
