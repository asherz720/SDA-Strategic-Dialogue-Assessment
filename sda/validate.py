"""Validate a user-parsed annotation table against the SDA schema.

The point of this module: you can run *any* model and parse its structured
output *however you like*. Before scoring, run :func:`validate` to confirm your
parsed table has the columns and exact value ranges SDA expects. It is the
single gatekeeper for input correctness — the scoring code assumes a table that
has passed here, so it does no defensive type-casting of its own.

:func:`validate` **warns** about problems and reports the offending rows rather
than crashing, so you see everything wrong at once::

    import pandas as pd
    from sda.validate import validate

    df = pd.read_csv("my_model_annotations.csv")
    report = validate(df)        # emits warnings
    if not report.ok:
        print(report.bad_rows)   # {column: [offending row indices]}
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

from .schema import CORE_FIELDS, OPTIONAL_FIELDS, REQUIRED_FIELDS


@dataclass
class ValidationReport:
    """Outcome of :func:`validate`.

    ``ok`` is True when there are no errors (missing required columns, or
    out-of-range / wrong-type / NaN values in required columns). Warnings alone
    do not make ``ok`` False.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # column -> sorted list of row indices with invalid values
    bad_rows: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def issues(self) -> list[str]:
        return self.errors + self.warnings

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "✅ Validation passed: table conforms to the SDA schema."
        lines = [f"❌ {e}" for e in self.errors]
        lines += [f"⚠️  {w}" for w in self.warnings]
        return "\n".join(lines)


def _check_range(df, report, col, allowed, *, required):
    """Flag values outside ``allowed`` (NaNs handled separately by caller)."""
    invalid_mask = ~df[col].isin(allowed) & df[col].notna()
    bad = df.index[invalid_mask].tolist()
    if not bad:
        return
    seen = sorted(set(df.loc[bad, col].tolist()), key=str)[:8]
    msg = (f"Column '{col}' has {len(bad)} value(s) outside {sorted(allowed, key=str)} "
           f"(e.g. {seen}); first rows: {bad[:10]}.")
    if required:
        report.bad_rows.setdefault(col, [])
        report.bad_rows[col] = sorted(set(report.bad_rows[col] + bad))
        report.errors.append(msg)
    else:
        report.warnings.append("Optional " + msg[0].lower() + msg[1:])


def validate(df: pd.DataFrame, *, emit_warnings: bool = True) -> ValidationReport:
    """Check ``df`` against the SDA schema.

    Values must already be the right type (e.g. integers, not the strings
    ``"2"``); mistyped values are reported as out-of-range so you fix your
    parser rather than relying on silent coercion. Returns a
    :class:`ValidationReport`; when ``emit_warnings`` is True, also surfaces each
    issue through :mod:`warnings`.
    """
    report = ValidationReport()

    # 1) required columns present?
    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            report.errors.append(f"Missing required column: '{col}'.")

    # 2) value-range checks on required columns we do have
    for col, allowed in CORE_FIELDS.items():
        if col in df.columns and allowed is not None:
            _check_range(df, report, col, allowed, required=True)

    # 3) NaNs in required columns
    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            continue
        rows = df.index[df[col].isna()].tolist()
        if rows:
            report.bad_rows.setdefault(col, [])
            report.bad_rows[col] = sorted(set(report.bad_rows[col] + rows))
            report.errors.append(f"Column '{col}' has {len(rows)} missing value(s); first rows: {rows[:10]}.")

    # 4) optional-column range checks -> warnings only
    for col, allowed in OPTIONAL_FIELDS.items():
        if col in df.columns and allowed is not None:
            _check_range(df, report, col, allowed, required=False)

    # 5) duplicate merge keys -> warning (correlation merges on 'question')
    if "question" in df.columns:
        dupes = int(df["question"].duplicated().sum())
        if dupes:
            report.warnings.append(
                f"'question' has {dupes} duplicate value(s); correlation merges on it, "
                "so duplicates may distort matches."
            )

    if emit_warnings:
        for msg in report.issues:
            warnings.warn(msg, stacklevel=2)

    return report
