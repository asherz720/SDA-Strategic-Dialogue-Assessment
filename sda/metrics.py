"""Compute the SDA metrics from validated annotations.

Public entry point is :func:`score`, which computes NRA (Normalized Relative
Advantage) and BaT / PaT / NRBaT from a table that has passed
:func:`sda.validate.validate` (so no defensive type-casting is done here).

The metrics are driven by commitment, manner and relevance; truthfulness
(``quality_rate``) is recorded but does not enter the scores by design.
"""

from __future__ import annotations

import numpy as np


def get_NRA(df):
    """Cumulative running tally of who is winning, per turn (in place)."""
    df['NRA'] = 0.0

    # Handle case with no annotator column (single annotator / one model run)
    if 'annotator' not in df.columns:
        for idx, row in df.iterrows():
            prev_rows = df[df.index <= idx]
            witness_count = len(prev_rows[prev_rows['outcome_value'] == 'Witness'])
            questioner_count = len(prev_rows[prev_rows['outcome_value'] == 'Questioner'])
            total_len = len(prev_rows)
            df.loc[idx, 'NRA'] = (witness_count - questioner_count) / total_len
    else:
        for annotator in df['annotator'].unique():
            annotator_data = df[df['annotator'] == annotator]
            for idx, row in annotator_data.iterrows():
                prev_rows = annotator_data[annotator_data.index <= idx]
                witness_count = len(prev_rows[prev_rows['outcome_value'] == 'Witness'])
                questioner_count = len(prev_rows[prev_rows['outcome_value'] == 'Questioner'])
                total_len = len(prev_rows)
                df.loc[idx, 'NRA'] = (witness_count - questioner_count) / total_len


def _bat_pat_for_row(row):
    """BaT/PaT contribution of a single turn from its rated fields."""
    pat = 0
    bat = 0
    cv = row['Commitment_value']
    # Base commitment value
    if cv == 2:      # beneficial
        bat += 1
    elif cv == 3:    # neutral
        bat += 0.5
    elif cv == 1:    # detrimental
        pat += 1
    elif cv == 4:    # none
        pat += 0.5

    # Manner and relevance: violating either (rate >= 3) erodes a beneficial
    # commitment and softens a detrimental one.
    if row['manner_rate'] >= 3:
        pat += 0.4 * (1 if cv == 2 else 0.5 if cv == 3 else 0)
        bat += 0.4 * (1 if cv == 1 else 0)
    if row['relevance_rate'] >= 3:
        pat += 0.4 * (1 if cv == 2 else 0.5 if cv == 3 else 0)
        bat += 0.4 * (1 if cv == 1 else 0)

    return bat, pat


def _znorm_block(df, block):
    """Cumulative sums, z-scores and NRBaT for one annotator block.

    ``block`` is the sub-frame for one annotator (or the whole frame in the
    single-annotator case). Results are written back into ``df`` by label.
    """
    index = block.index
    cum_bats, cum_pats = [], []
    for idx in index:
        prev_rows = block.loc[:idx]   # cumulative rows up to this one, within block
        bat_sum = prev_rows['bat'].sum()
        pat_sum = prev_rows['pat'].sum()
        cum_bats.append(bat_sum)
        cum_pats.append(pat_sum)
        df.loc[idx, 'bat_cumsum'] = bat_sum
        df.loc[idx, 'pat_cumsum'] = pat_sum

    bat_z = (np.array(cum_bats) - np.mean(cum_bats)) / np.std(cum_bats, ddof=0)
    pat_z = (np.array(cum_pats) - np.mean(cum_pats)) / np.std(cum_pats, ddof=0)
    df.loc[index, 'Z_BaT'] = bat_z
    df.loc[index, 'Z_PaT'] = pat_z

    for i, idx in enumerate(index):
        numerator = bat_z[i] - pat_z[i]
        denominator = bat_z[i] + pat_z[i]
        if denominator != 0:
            df.loc[idx, 'ZNRBaT'] = numerator / denominator
            df.loc[idx, 'net_ZNRBaT'] = numerator
        else:
            df.loc[idx, 'ZNRBaT'] = 0
            df.loc[idx, 'net_ZNRBaT'] = 0


def get_NRBaT(df):
    """Compute BaT, PaT, their z-scored cumulative forms and NRBaT (in place)."""
    df['bat'] = 0.0
    df['pat'] = 0.0

    if 'annotator' not in df.columns:
        for idx, row in df.iterrows():
            bat, pat = _bat_pat_for_row(row)
            df.loc[idx, 'bat'] = bat
            df.loc[idx, 'pat'] = pat

        # Consistency penalty: an inconsistent turn is penalised by prior benefit.
        for idx, row in df.iterrows():
            if row['consistency_value'] == 1:
                prev_bat_sum = df[df.index <= idx]['bat'].sum()
                df.loc[idx, 'pat'] = df['pat'][idx] + 0.2 * prev_bat_sum

        _znorm_block(df, df)
    else:
        for annotator in df['annotator'].unique():
            annotator_data = df[df['annotator'] == annotator]
            for idx, row in annotator_data.iterrows():
                bat, pat = _bat_pat_for_row(row)
                df.loc[idx, 'bat'] = bat
                df.loc[idx, 'pat'] = pat

        for annotator in df['annotator'].unique():
            annotator_data = df[df['annotator'] == annotator]
            for idx, row in annotator_data.iterrows():
                if row['consistency_value'] == 1:
                    prev_bat_sum = annotator_data[annotator_data.index <= idx]['bat'].sum()
                    df.loc[idx, 'pat'] = df['pat'][idx] + 0.2 * prev_bat_sum

        for annotator in df['annotator'].unique():
            _znorm_block(df, df[df['annotator'] == annotator])


def score(df, *, copy=True):
    """Compute all SDA metrics (NRA, BaT, PaT, NRBaT) for a validated table.

    Returns the per-turn scored DataFrame (a copy by default). Run
    :func:`sda.validate.validate` first; this function assumes clean input.
    Write it out with ``scored.to_csv(path, index=False)``.
    """
    if copy:
        df = df.copy()
    get_NRA(df)
    get_NRBaT(df)
    return df
