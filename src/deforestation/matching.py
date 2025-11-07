"""Core propensity-score matching utilities.

The original repository implemented these ideas in standalone scripts. This module
provides a reusable, well-tested API that exposes the key matching steps:

* fit a propensity-score model
* perform calipered nearest-neighbour matching
* compute balance diagnostics and treatment effects

The functions are written to operate on in-memory ``pandas`` objects only so they are
simple to unit test and easy to integrate into notebooks, CLIs, or pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class MatchingError(RuntimeError):
    """Raised when matching fails (e.g. no units fall within the caliper)."""


@dataclass
class MatchSummary:
    """Container for the main outputs of the matching pipeline."""

    matched_data: pd.DataFrame
    """Rows for matched treated and control units, annotated with ``match_id``."""

    matches: List[Tuple[Hashable, Hashable]]
    """Pairs of (treated_index, control_index) that were matched."""

    att: float
    """Average treatment effect on the treated calculated on ``matched_data``."""

    balance: pd.DataFrame
    """Covariate balance table after matching."""

    auc: float
    """Area under the ROC curve for the propensity-score model."""


def estimate_propensity(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model: Optional[BaseEstimator] = None,
) -> Tuple[pd.Series, BaseEstimator, float]:
    """Fit a propensity-score model.

    Parameters
    ----------
    X:
        Covariate matrix.
    y:
        Binary treatment indicator (1 = treated, 0 = control).
    model:
        Optional scikit-learn estimator. Defaults to ``LogisticRegression`` with a
        high iteration cap for convergence.

    Returns
    -------
    propensity_scores, fitted_model, auc
        Predicted probabilities for the treated class, the fitted estimator, and the
        model AUC (useful for quick diagnostics).
    """

    if model is None:
        model = LogisticRegression(max_iter=5_000)

    fitted_model = model.fit(X, y)
    propensity = fitted_model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, propensity)
    propensity_series = pd.Series(propensity, index=X.index, name="propensity_score")
    return propensity_series, fitted_model, auc


def match_nearest_neighbors(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    score_col: str,
    caliper: float = 0.1,
    allow_replacement: bool = False,
) -> Tuple[pd.DataFrame, List[Tuple[Hashable, Hashable]]]:
    """Perform one-to-one nearest-neighbour matching on the propensity score.

    Parameters
    ----------
    df:
        Data with treatment indicator and propensity scores.
    treatment_col:
        Name of the column with the binary treatment indicator (1 = treated).
    score_col:
        Name of the column containing the propensity score.
    caliper:
        Maximum allowed absolute difference in propensity score between matched
        treated and control units.
    allow_replacement:
        If ``True`` control units can be matched multiple times.

    Returns
    -------
    matched, pairs
        ``matched`` contains the rows from ``df`` that were matched and includes a
        ``match_id`` column. ``pairs`` enumerates the underlying index matches.
    """

    treated_mask = df[treatment_col] == 1
    treated = df.loc[treated_mask].copy()
    control = df.loc[~treated_mask].copy()

    if treated.empty or control.empty:
        raise MatchingError("Both treated and control units are required for matching.")

    available_control = control.copy()
    matches: List[Tuple[Hashable, Hashable]] = []

    for treated_idx, treated_row in treated.iterrows():
        score_diff = (available_control[score_col] - treated_row[score_col]).abs()
        if score_diff.empty:
            break
        nearest_idx = score_diff.idxmin()
        nearest_diff = score_diff.loc[nearest_idx]
        if nearest_diff > caliper:
            continue
        matches.append((treated_idx, nearest_idx))
        if not allow_replacement:
            available_control = available_control.drop(index=nearest_idx)

    if not matches:
        raise MatchingError("No matches found within the specified caliper.")

    matched_frames = []
    for match_id, (treated_idx, control_idx) in enumerate(matches):
        subset = df.loc[[treated_idx, control_idx]].copy()
        subset["match_id"] = match_id
        matched_frames.append(subset)

    matched = pd.concat(matched_frames, axis=0).sort_values("match_id")
    return matched, matches


def compute_att(
    matched_df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
) -> float:
    """Compute the average treatment effect on the treated (ATT)."""

    treated_outcomes = matched_df.loc[matched_df[treatment_col] == 1, outcome_col]
    control_outcomes = matched_df.loc[matched_df[treatment_col] == 0, outcome_col]
    if treated_outcomes.empty or control_outcomes.empty:
        raise MatchingError("ATT requires both treated and control outcomes.")
    return float(treated_outcomes.mean() - control_outcomes.mean())


def covariate_balance(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Compute means and standardised differences for each covariate."""

    results = []
    groups = df.groupby(treatment_col)
    for covariate in covariates:
        treated_mean = groups[covariate].mean().get(1, np.nan)
        control_mean = groups[covariate].mean().get(0, np.nan)
        treated_var = groups[covariate].var(ddof=1).get(1, np.nan)
        control_var = groups[covariate].var(ddof=1).get(0, np.nan)
        if np.isnan(treated_var) or np.isnan(control_var):
            pooled_sd = np.nan
        else:
            pooled_sd = float(np.sqrt((treated_var + control_var) / 2))
        std_diff = (
            (treated_mean - control_mean) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan
        )
        results.append(
            {
                "covariate": covariate,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "std_diff": std_diff,
            }
        )
    return pd.DataFrame(results)


def run_matching_pipeline(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariates: Sequence[str],
    caliper: float = 0.1,
    allow_replacement: bool = False,
    model: Optional[BaseEstimator] = None,
) -> MatchSummary:
    """High-level convenience wrapper executing the full matching workflow."""

    X = df[covariates]
    y = df[treatment_col]
    propensity_scores, fitted_model, auc = estimate_propensity(X, y, model=model)
    df = df.copy()
    df["propensity_score"] = propensity_scores
    matched, matches = match_nearest_neighbors(
        df,
        treatment_col=treatment_col,
        score_col="propensity_score",
        caliper=caliper,
        allow_replacement=allow_replacement,
    )
    att = compute_att(matched, treatment_col=treatment_col, outcome_col=outcome_col)
    balance = covariate_balance(matched, treatment_col=treatment_col, covariates=covariates)
    return MatchSummary(matched, matches, att, balance, auc)
