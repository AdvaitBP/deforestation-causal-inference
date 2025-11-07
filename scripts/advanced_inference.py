import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# -----------------------------
# Randomization Inference for ATT
# -----------------------------
def randomization_inference(df, outcome_col="lossyear_mean", n_perm=1000):
    """
    Perform randomization inference on matched groups.
    Returns:
      - att_obs: observed ATT
      - p_val: p-value from RI
      - perm_atts: array of permuted ATTs
      - group_diffs: array of observed paired differences
    """
    group_diffs = []
    for group_id, group in df.groupby("matched_id"):
        if group["is_protected"].nunique() < 2:
            continue
        treated = group[group["is_protected"] == 1][outcome_col].mean()
        control = group[group["is_protected"] == 0][outcome_col].mean()
        group_diffs.append(treated - control)
    group_diffs = np.array(group_diffs)
    att_obs = group_diffs.mean()
    
    perm_atts = []
    for _ in range(n_perm):
        flips = np.random.choice([1, -1], size=len(group_diffs))
        perm_atts.append((group_diffs * flips).mean())
    perm_atts = np.array(perm_atts)
    
    p_val = np.mean(np.abs(perm_atts) >= np.abs(att_obs))
    return att_obs, p_val, perm_atts, group_diffs

# -----------------------------
# Rosenbaum Sensitivity Analysis
# -----------------------------
def sensitivity_ri(diffs, observed_att, gamma, num_sims=20000, seed=42):
    """
    Rosenbaum-style sensitivity via biased sign-flip randomization.
    Γ = gamma ≥ 1 controls bias: P(sign=+1)=Γ/(1+Γ).
    Returns two-sided p-value under this biased randomization.
    """
    rng = np.random.default_rng(seed)
    p = gamma / (1 + gamma)
    flips = rng.random((num_sims, len(diffs))) < p
    sim_atts = (flips * diffs[np.newaxis, :] + (~flips) * (-diffs)[np.newaxis, :]).mean(axis=1)
    pval = np.mean(np.abs(sim_atts) >= abs(observed_att))
    return pval

# -----------------------------
# Doubly Robust Estimator using Outcome Regression
# -----------------------------
def doubly_robust_estimator(df, outcome_col="lossyear_mean", covariate_cols=None):
    """
    Fit an OLS regression on matched data with treatment + covariates.
    Returns (coef, se, p_value) for the treatment effect.
    """
    if covariate_cols is None:
        covariate_cols = []
    X = df[["is_protected"] + covariate_cols].copy()
    X = sm.add_constant(X)
    y = df[outcome_col]
    model = sm.OLS(y, X).fit(cov_type="HC3")
    coef = model.params["is_protected"]
    se = model.bse["is_protected"]
    p_value = model.pvalues["is_protected"]
    print(model.summary())
    return coef, se, p_value

# -----------------------------
# Plotting Function for Randomization Distribution
# -----------------------------
def plot_randomization_distribution(perm_atts, att_obs):
    plt.figure(figsize=(8,6))
    sns.histplot(perm_atts, bins=30, kde=True)
    plt.axvline(att_obs, color="red", linestyle="--", label=f"Observed ATT = {att_obs:.4f}")
    plt.xlabel("ATT")
    plt.ylabel("Frequency")
    plt.title("Randomization Inference Null Distribution")
    plt.legend()
    plt.show()

# -----------------------------
# Main Function: Advanced Inference
# -----------------------------
def main():
    # Load matched data
    matched_fp = os.path.join("..", "data", "processed", "matched_data.csv")
    df = pd.read_csv(matched_fp)
    
    # Check required columns
    for col in ["matched_id", "is_protected", "lossyear_mean"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in matched_data.csv")
    
    # 1. Randomization Inference
    print("=== Randomization Inference ===")
    att_obs, p_val, perm_atts, group_diffs = randomization_inference(df, "lossyear_mean", n_perm=1000)
    print(f"Observed ATT = {att_obs:.4f}")
    print(f"RI p-value = {p_val:.4f}")
    plot_randomization_distribution(perm_atts, att_obs)
    
    # 2. Rosenbaum Sensitivity Analysis
    print("\n=== Rosenbaum Sensitivity Analysis ===")
    for G in [1.0, 1.2, 1.5, 2.0]:
        p_sens = sensitivity_ri(group_diffs, att_obs, gamma=G)
        print(f"Gamma = {G:.1f} -> p = {p_sens:.3f}")
    
    # 3. Doubly Robust Estimation
    print("\n=== Doubly Robust Estimation ===")
    covs = [c for c in ["pop_density", "road_dist", "slope", "elevation", "precip_mean", "treecover2000_mean"] if c in df.columns]
    print("Covariates used:", covs)
    coef, se, p = doubly_robust_estimator(df, "lossyear_mean", covs)
    print(f"DR estimate = {coef:.4f}, SE = {se:.4f}, p = {p:.4f}")

if __name__ == "__main__":
    main()
