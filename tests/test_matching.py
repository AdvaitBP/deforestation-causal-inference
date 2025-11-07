import numpy as np
import pandas as pd
import pytest

from deforestation.matching import (
    MatchingError,
    compute_att,
    covariate_balance,
    estimate_propensity,
    match_nearest_neighbors,
    run_matching_pipeline,
)


def make_synthetic_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    road_dist = rng.normal(5, 1.5, size=n)
    pop_density = rng.normal(50, 10, size=n)
    logits = -1 + 0.3 * road_dist - 0.02 * pop_density
    p_treat = 1 / (1 + np.exp(-logits))
    treatment = rng.binomial(1, p_treat)
    baseline = 10 - 0.1 * road_dist + 0.05 * pop_density
    att_true = -1.25
    outcome = baseline + att_true * treatment + rng.normal(0, 0.5, size=n)
    return pd.DataFrame(
        {
            "road_dist": road_dist,
            "pop_density": pop_density,
            "is_protected": treatment,
            "lossyear_mean": outcome,
        }
    )


def test_estimate_propensity_returns_probabilities():
    df = make_synthetic_data()
    propensity, model, auc = estimate_propensity(
        df[["road_dist", "pop_density"]], df["is_protected"]
    )
    assert ((propensity >= 0) & (propensity <= 1)).all()
    assert 0.5 < auc < 1.0


def test_match_nearest_neighbors_respects_caliper():
    df = pd.DataFrame(
        {
            "propensity_score": [0.9, 0.85, 0.1, 0.05],
            "is_protected": [1, 1, 0, 0],
            "road_dist": [1, 2, 3, 4],
            "pop_density": [10, 12, 30, 35],
        },
        index=["t1", "t2", "c1", "c2"],
    )
    with pytest.raises(MatchingError):
        match_nearest_neighbors(
            df,
            treatment_col="is_protected",
            score_col="propensity_score",
            caliper=0.1,
        )


def test_run_matching_pipeline_produces_att_close_to_true_effect():
    df = make_synthetic_data(400)
    summary = run_matching_pipeline(
        df,
        treatment_col="is_protected",
        outcome_col="lossyear_mean",
        covariates=["road_dist", "pop_density"],
        caliper=0.25,
    )
    assert summary.matched_data["match_id"].nunique() > 10
    assert summary.balance.shape[0] == 2
    assert summary.att < 0  # protective effect


def test_covariate_balance_returns_expected_columns():
    df = make_synthetic_data(120)
    propensity, _, _ = estimate_propensity(df[["road_dist", "pop_density"]], df["is_protected"])
    df = df.assign(propensity_score=propensity)
    matched, _ = match_nearest_neighbors(
        df,
        treatment_col="is_protected",
        score_col="propensity_score",
        caliper=0.3,
    )
    balance = covariate_balance(
        matched,
        treatment_col="is_protected",
        covariates=["road_dist", "pop_density"],
    )
    assert set(balance.columns) == {"covariate", "treated_mean", "control_mean", "std_diff"}
    assert balance.shape[0] == 2


def test_compute_att_matches_manual_difference():
    df = make_synthetic_data(120)
    propensity, _, _ = estimate_propensity(df[["road_dist", "pop_density"]], df["is_protected"])
    df = df.assign(propensity_score=propensity)
    matched, _ = match_nearest_neighbors(
        df,
        treatment_col="is_protected",
        score_col="propensity_score",
        caliper=0.3,
    )
    att = compute_att(matched, treatment_col="is_protected", outcome_col="lossyear_mean")
    treated = matched.loc[matched["is_protected"] == 1, "lossyear_mean"]
    control = matched.loc[matched["is_protected"] == 0, "lossyear_mean"]
    assert att == pytest.approx(float(treated.mean() - control.mean()))
