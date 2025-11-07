"""Utilities for causal inference on deforestation data."""

from .matching import (
    MatchingError,
    MatchSummary,
    compute_att,
    covariate_balance,
    estimate_propensity,
    match_nearest_neighbors,
    run_matching_pipeline,
)

__all__ = [
    "MatchingError",
    "MatchSummary",
    "compute_att",
    "covariate_balance",
    "estimate_propensity",
    "match_nearest_neighbors",
    "run_matching_pipeline",
]
