import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from rasterstats import zonal_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import logit
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm

# -----------------------------
# Step 0: Utility Functions
# -----------------------------
def compute_fraction_lost(grid, cover2000_col="treecover2000", lost_col="lost_area"):
    """
    If your grid has columns:
      - treecover2000: total forest area in 2000 (in hectares or another unit)
      - lost_area: total forest area lost by 2020
    Then fraction_lost = lost_area / treecover2000.

    If you only have 'lossyear' rasters, you'll need to first compute how many pixels lost in each cell.
    This function assumes you already have a 'lost_area' column from zonal_stats on the Hansen 'loss' band.
    """
    grid["fraction_lost"] = grid[lost_col] / grid[cover2000_col]
    # If fraction_lost > 1 or < 0, clamp or drop as needed
    grid["fraction_lost"] = grid["fraction_lost"].clip(lower=0, upper=1)
    return grid

def standardized_mean_diff(treated_vals, control_vals):
    """
    Compute standardized mean difference (SMD) for a single covariate:
      SMD = (mean_t - mean_c) / sqrt((var_t + var_c)/2).
    """
    mean_t = np.mean(treated_vals)
    mean_c = np.mean(control_vals)
    var_t = np.var(treated_vals, ddof=1)
    var_c = np.var(control_vals, ddof=1)
    smd = (mean_t - mean_c) / np.sqrt((var_t + var_c) / 2)
    return smd

def love_plot(smd_before, smd_after):
    """
    Create a Love plot comparing standardized mean differences before and after matching.
      smd_before: dict of {covariate: smd_value}
      smd_after: dict of {covariate: smd_value}
    """
    covariates = list(smd_before.keys())
    data = []
    for cov in covariates:
        data.append((cov, smd_before[cov], "Before Matching"))
        data.append((cov, smd_after[cov], "After Matching"))
    df_plot = pd.DataFrame(data, columns=["Covariate", "SMD", "Stage"])
    
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df_plot, x="SMD", y="Covariate", hue="Stage", dodge=True)
    plt.axvline(x=0, color="grey", linestyle="--")
    plt.title("Love Plot: Standardized Mean Differences Before vs. After Matching")
    plt.show()

# -----------------------------
# Step 1: Load & Prepare Data
# -----------------------------
def load_and_prepare_data():
    """
    We assume you have a GeoDataFrame with:
      - is_protected (1/0)
      - slope, elevation, precip, pop_density, road_dist, treecover2000, lost_area
      - geometry (optional, not strictly needed for matching)
    We'll compute fraction_lost as the outcome.
    """
    data_fp = os.path.join("..", "data", "processed", "grid_stats_with_protection.geojson")
    gdf = gpd.read_file(data_fp)
    df = pd.DataFrame(gdf.drop(columns="geometry"))  # convert to DataFrame
    
    # Compute fraction of forest lost
    df = compute_fraction_lost(df, cover2000_col="treecover2000", lost_col="lost_area")
    
    # Drop rows with invalid fraction_lost or missing data
    df = df.dropna(subset=["fraction_lost", "slope", "elevation", "precip", 
                           "pop_density", "road_dist", "is_protected"])
    
    return df

# -----------------------------
# Step 2: Define Covariates & Outcome
# -----------------------------
def define_covariates_and_outcome(df):
    """
    Covariates: slope, elevation, precipitation, population density, road distance, initial tree cover, etc.
    Outcome: fraction_lost
    Treatment: is_protected
    """
    covariate_cols = ["slope", "elevation", "precip", "pop_density", "road_dist"]
    # Add "treecover2000" if you want to treat that as a covariate as well.
    
    X = df[covariate_cols]
    y = df["is_protected"]
    outcome = df["fraction_lost"]
    
    return X, y, outcome, covariate_cols

# -----------------------------
# Step 3: Propensity Score Estimation
# -----------------------------
def estimate_propensity(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    propensity = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, propensity)
    print(f"Propensity score model AUC: {auc:.3f}")
    return propensity, model

def plot_propensity_distribution(df, covariate_cols):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="propensity_score", hue="is_protected", element="step", stat="density", bins=30)
    plt.title("Propensity Score Distribution by Protection Status")
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.legend(title="Protected", labels=["Unprotected", "Protected"])
    plt.show()

# -----------------------------
# Step 4: Matching (Multi-to-One)
# -----------------------------
def multi_match(df, ratio=3, caliper=0.1, replacement=False):
    """
    Multi-to-one matching: for each treated unit, we find 'ratio' control units
    with the closest propensity scores within the caliper. 
    If replacement=True, the same control can be used multiple times.
    """
    treated = df[df["is_protected"] == 1].copy()
    control = df[df["is_protected"] == 0].copy()
    
    matches = []
    if not replacement:
        # We'll drop matched controls so they can't be reused
        ctrl_pool = control.copy()
    else:
        # Keep the entire control pool for reuse
        ctrl_pool = control
    
    for t_idx, row in treated.iterrows():
        t_score = row["propensity_score"]
        ctrl_pool["abs_diff"] = (ctrl_pool["propensity_score"] - t_score).abs()
        # Filter to within caliper
        within_caliper = ctrl_pool[ctrl_pool["abs_diff"] <= caliper].copy()
        if len(within_caliper) == 0:
            continue  # no matches
        # Sort by absolute difference
        within_caliper = within_caliper.sort_values("abs_diff")
        # Take the top 'ratio' controls
        selected = within_caliper.head(ratio)
        # Record matches
        for c_idx in selected.index:
            matches.append((t_idx, c_idx))
        if not replacement:
            # Remove matched controls from pool
            ctrl_pool = ctrl_pool.drop(selected.index)
    
    # Build matched DataFrame
    matched_rows = []
    for (t_idx, c_idx) in matches:
        t_row = df.loc[t_idx].copy()
        c_row = df.loc[c_idx].copy()
        t_row["matched_id"] = t_idx
        c_row["matched_id"] = t_idx
        matched_rows.append(t_row)
        matched_rows.append(c_row)
    matched_df = pd.DataFrame(matched_rows)
    
    # Print summary
    unique_treated = len(set([m[0] for m in matches]))
    print(f"Multi-to-one matching complete. ratio={ratio}, caliper={caliper}, replacement={replacement}")
    print(f"Number of unique treated units matched: {unique_treated}")
    print(f"Total matched rows: {len(matched_rows)} (this includes treated + controls).")
    return matched_df

# -----------------------------
# Step 5: Balance Diagnostics (SMD & Love Plot)
# -----------------------------
def balance_diagnostics(df_full, df_matched, covariate_cols):
    """
    Compute SMD for each covariate before and after matching, and create a Love plot.
    """
    # Pre-matching
    smd_before = {}
    for var in covariate_cols:
        treated_vals = df_full[df_full["is_protected"] == 1][var]
        control_vals = df_full[df_full["is_protected"] == 0][var]
        smd_before[var] = standardized_mean_diff(treated_vals, control_vals)
    
    # Post-matching
    smd_after = {}
    for var in covariate_cols:
        treated_vals = df_matched[(df_matched["is_protected"] == 1)][var]
        control_vals = df_matched[(df_matched["is_protected"] == 0)][var]
        smd_after[var] = standardized_mean_diff(treated_vals, control_vals)
    
    # Love plot
    love_plot(smd_before, smd_after)

# -----------------------------
# Step 6: Compute ATT + Confidence Interval via Bootstrap
# -----------------------------
def compute_ATT_bootstrap(matched_df, outcome_col="fraction_lost", n_boot=500, alpha=0.05):
    """
    Compute ATT = E[Y|treated] - E[Y|control] in the matched sample,
    plus a bootstrap confidence interval.
    """
    # Observed ATT
    treated_outcomes = matched_df[matched_df["is_protected"] == 1][outcome_col]
    control_outcomes = matched_df[matched_df["is_protected"] == 0][outcome_col]
    att_obs = treated_outcomes.mean() - control_outcomes.mean()
    
    # Bootstrap
    rng = np.random.default_rng(42)
    atts = []
    # matched_df is grouped by matched_id; we can sample matched_id's
    unique_ids = matched_df["matched_id"].unique()
    for _ in tqdm(range(n_boot), desc="Bootstrapping ATT"):
        # sample matched_id with replacement
        sample_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        sample_rows = matched_df[matched_df["matched_id"].isin(sample_ids)]
        t_vals = sample_rows[sample_rows["is_protected"] == 1][outcome_col]
        c_vals = sample_rows[sample_rows["is_protected"] == 0][outcome_col]
        if len(t_vals) == 0 or len(c_vals) == 0:
            # skip if empty
            continue
        atts.append(t_vals.mean() - c_vals.mean())
    
    # Confidence interval
    lower = np.percentile(atts, 100 * alpha / 2)
    upper = np.percentile(atts, 100 * (1 - alpha / 2))
    
    print(f"Observed ATT: {att_obs:.4f}")
    print(f"{int((1-alpha)*100)}% CI: [{lower:.4f}, {upper:.4f}] (bootstrap, n={len(atts)})")
    return att_obs, lower, upper

# -----------------------------
# Main: Putting It All Together
# -----------------------------
def main():
    # 1. Load data and compute fraction_lost outcome
    df = load_and_prepare_data()
    
    # 2. Define covariates & outcome
    X, y, outcome, covariate_cols = define_covariates_and_outcome(df)
    
    # 3. Estimate propensity scores
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    propensity = model.predict_proba(X)[:, 1]
    df["propensity_score"] = propensity
    auc = roc_auc_score(y, propensity)
    print(f"Propensity score model AUC: {auc:.3f}")
    
    # 4. Multi-to-one matching (ratio=3, caliper=0.1, no replacement)
    matched_df = multi_match(df, ratio=3, caliper=0.1, replacement=False)
    
    # 5. Balance diagnostics
    balance_diagnostics(df, matched_df, covariate_cols)
    
    # 6. Compute ATT + CI with bootstrap
    att_obs, ci_lower, ci_upper = compute_ATT_bootstrap(matched_df, outcome_col="fraction_lost", n_boot=500, alpha=0.05)
    
    # Save matched data
    out_fp = os.path.join("..", "data", "processed", "matched_data_improved.csv")
    matched_df.to_csv(out_fp, index=False)
    print("Matched data saved to:", out_fp)

if __name__ == "__main__":
    main()
