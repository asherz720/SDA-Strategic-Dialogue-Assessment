"""SDA — Strategic Dialogue Assessment.

A small toolkit to evaluate how well a model judges strategic, non-cooperative
dialogue. You bring a table of parsed model annotations; SDA validates it,
computes the BaT / PaT / NRBaT metrics, and correlates them with human ratings.

Typical flow::

    import pandas as pd
    from sda import validate, score, compare_human_llm

    model_df = pd.read_csv("my_model_annotations.csv")  # your parsed output
    report = validate(model_df)          # 1. check format, warn on problems
    scored = score(model_df)             # 2. derive binaries + BaT/PaT/NRBaT
    human_df = pd.read_csv("human annotations/WMT_D_annotations.csv")
    results = compare_human_llm(scored, human_df)        # 3. correlate
"""

from .agreement import compare_human_llm
from .metrics import get_NRA, get_NRBaT, score
from .validate import ValidationReport, validate

__all__ = [
    "validate",
    "ValidationReport",
    "score",
    "get_NRA",
    "get_NRBaT",
    "compare_human_llm",
]
