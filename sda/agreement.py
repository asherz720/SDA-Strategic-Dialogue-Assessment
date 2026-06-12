"""Correlation and inter-rater agreement between model and human annotations.

Model and human tables are merged on ``question``; humans may have multiple
annotators (a per-annotator score is averaged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa


def spearman_vs_humans(llm_df, human_df, value, *, verbose=True):
    """Mean Spearman correlation of ``value`` between the model and each human."""
    correlations = []
    if verbose:
        print(f"Correlations between model and human annotators for {value}:")
    for annotator in human_df['annotator'].unique():
        annotator_df = human_df[human_df['annotator'] == annotator]
        merged_df = pd.merge(llm_df[['question', value]],
                             annotator_df[['question', value]],
                             on='question', suffixes=('_gpt', '_human'))
        merged_df = merged_df.dropna(subset=[f'{value}_gpt', f'{value}_human'])
        correlation, p_value = scipy.stats.spearmanr(merged_df[f'{value}_gpt'], merged_df[f'{value}_human'])
        correlations.append(correlation)
        if verbose:
            print(f"  {annotator}: {correlation:.3f} (p={p_value:.2e})")
    return float(np.mean(correlations))


def cohen_vs_humans(llm_df, human_df, value, *, verbose=True):
    """Mean Cohen's kappa of ``value`` between the model and each human."""
    kappas = []
    if verbose:
        print(f"\nCohen's kappa between model and human annotators for {value}:")
    for annotator in human_df['annotator'].unique():
        annotator_df = human_df[human_df['annotator'] == annotator]
        merged_df = pd.merge(llm_df[['question', value]],
                             annotator_df[['question', value]],
                             on='question', suffixes=('_gpt', '_human'))
        merged_df = merged_df.dropna(subset=[f'{value}_gpt', f'{value}_human'])
        # binarise the categorical outcome before kappa
        if value == 'outcome_value':
            merged_df[f'{value}_gpt'] = (merged_df[f'{value}_gpt'] == 'Witness').astype(int)
            merged_df[f'{value}_human'] = (merged_df[f'{value}_human'] == 'Witness').astype(int)
        kappa = cohen_kappa_score(merged_df[f'{value}_gpt'], merged_df[f'{value}_human'])
        kappas.append(kappa)
        if verbose:
            print(f"  {annotator}: kappa={kappa:.3f}")
    return float(np.mean(kappas))


def randolph_vs_humans(llm_df, human_df, value, *, binarize=False, verbose=True):
    """Mean Randolph's (free-marginal) kappa of ``value`` vs each human.

    With ``binarize=True`` the 1-4 ``value`` is collapsed to "maxim violated"
    (``rate > 2``) before computing agreement — used for relevance and manner.
    """
    kappas = []
    if verbose:
        print(f"\nRandolph's kappa between model and human annotators for {value}:")
    for annotator in human_df['annotator'].unique():
        annotator_df = human_df[human_df['annotator'] == annotator]
        merged_df = pd.merge(llm_df[['question', value]],
                             annotator_df[['question', value]],
                             on='question', suffixes=('_gpt', '_human'))
        merged_df = merged_df.dropna(subset=[f'{value}_gpt', f'{value}_human']).reset_index(drop=True)
        if binarize:
            merged_df[f'{value}_gpt'] = (merged_df[f'{value}_gpt'] > 2).astype(int)
            merged_df[f'{value}_human'] = (merged_df[f'{value}_human'] > 2).astype(int)
        agreement = (merged_df[f'{value}_gpt'] == merged_df[f'{value}_human']).mean() * 100
        # Map observed labels -> column indices (robust to any/non-contiguous codes).
        cats = sorted(set(merged_df[f'{value}_gpt']) | set(merged_df[f'{value}_human']))
        idx = {c: i for i, c in enumerate(cats)}
        ratings = np.zeros((len(merged_df), len(cats)))
        for i, row in merged_df.iterrows():
            ratings[i, idx[row[f'{value}_gpt']]] += 1
            ratings[i, idx[row[f'{value}_human']]] += 1
        kappa = fleiss_kappa(ratings, method='randolph')
        kappas.append(kappa)
        if verbose:
            print(f"  {annotator}: agreement={agreement:.1f}%, kappa={kappa:.3f}")
    return float(np.mean(kappas))


def pairwise_accuracy(llm_df, human_df, value):
    """Mean true-positive rate of ``value`` (both rate it 1) across humans."""
    rates = []
    for annotator in human_df['annotator'].unique():
        annotator_df = human_df[human_df['annotator'] == annotator]
        merged_df = pd.merge(llm_df[['question', value]],
                             annotator_df[['question', value]],
                             on='question', suffixes=('_gpt', '_human'))
        merged_df = merged_df.dropna(subset=[f'{value}_gpt', f'{value}_human'])
        if len(merged_df) == 0:
            continue
        true_positives = ((merged_df[f'{value}_gpt'] == 1) & (merged_df[f'{value}_human'] == 1)).sum()
        actual_positives = (merged_df[f'{value}_human'] == 1).sum()
        rates.append(true_positives / actual_positives if actual_positives > 0 else 0)
    return float(np.mean(rates)) if rates else float('nan')


# The metrics reported in summary tables, in order. bat/pat/nrbat are Spearman
# correlations; the rest are agreement (kappa, or accuracy for consistency). Keys
# are compare_human_llm's keys; values are the report column names. (outcome and
# nra are still computed by compare_human_llm but omitted from the summary.)
REPORT_METRICS = {
    'bat': 'bat', 'pat': 'pat', 'nrbat': 'nrbat',
    'commitment': 'commitment', 'relevance': 'relevance', 'manner': 'manner',
    'quality': 'quality', 'consistency': 'consistency',
}


def mean_across(df, label='witness', sum_cols=()):
    """Collapse a per-witness table to a single mean row.

    Numeric columns are averaged, except those in ``sum_cols`` (e.g. turn
    counts) which are summed. The label column records how many rows were
    aggregated.
    """
    num = df.select_dtypes('number')
    agg = {c: (num[c].sum() if c in sum_cols else num[c].mean()) for c in num.columns}
    out = pd.DataFrame([agg])
    out.insert(0, label, f"MEAN of {len(df)}")
    return out


def evaluate_dataset(pairs, *, level='mean', verbose=False):
    """Agreement-with-humans across many witnesses/trials.

    ``pairs`` is an iterable of ``(name, model_df, human_df)``. Returns a tidy
    DataFrame with one row per witness (``level='witness'``) or a single mean
    row (``level='mean'``). Write it with ``df.to_csv(path, index=False)``.
    """
    rows = []
    for name, model_df, human_df in pairs:
        res = compare_human_llm(model_df, human_df, verbose=verbose)
        row = {'witness': name}
        row.update({col: res[k] for k, col in REPORT_METRICS.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    return mean_across(df) if level == 'mean' else df


def compare_human_llm(llm_df, human_df, *, verbose=True):
    """Full comparison used in the paper.

    Returns a dict of the headline numbers: Spearman correlation for the
    continuous metrics (BaT, PaT, NRBaT, NRA) and kappa/accuracy for the
    categorical judgements.
    """
    results = {
        'bat': spearman_vs_humans(llm_df, human_df, 'bat', verbose=verbose),
        'pat': spearman_vs_humans(llm_df, human_df, 'pat', verbose=verbose),
        'nrbat': spearman_vs_humans(llm_df, human_df, 'net_ZNRBaT', verbose=verbose),
        'nra': spearman_vs_humans(llm_df, human_df, 'NRA', verbose=verbose),
        'outcome': cohen_vs_humans(llm_df, human_df, 'outcome_value', verbose=verbose),
        'commitment': cohen_vs_humans(llm_df, human_df, 'Commitment_value', verbose=verbose),
        'relevance': randolph_vs_humans(llm_df, human_df, 'relevance_rate', binarize=True, verbose=verbose),
        'manner': randolph_vs_humans(llm_df, human_df, 'manner_rate', binarize=True, verbose=verbose),
        'quality': randolph_vs_humans(llm_df, human_df, 'quality_rate', verbose=verbose),
        'consistency': pairwise_accuracy(llm_df, human_df, 'consistency_value'),
    }
    return results
