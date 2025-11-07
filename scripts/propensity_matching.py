import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from rasterstats import zonal_stats
from shapely.ops import nearest_points
from shapely.geometry import Point

# -----------------------------
# Step 1: Load Preprocessed Grid Data
# -----------------------------
def load_grid():
    # Load the grid with zonal statistics and protection indicator
    grid_fp = os.path.join("..", "data", "processed", "grid_stats_with_protection.geojson")
    grid = gpd.read_file(grid_fp)
    return grid

# -----------------------------
# Step 2: Add Real Covariates
# -----------------------------
def add_population_covariate(grid, pop_fp):
    """
    Compute mean population density for each grid cell using zonal_stats
    from the GPWv4 population raster.
    """
    print("Extracting population density...")
    stats = zonal_stats(grid.geometry, pop_fp, stats="mean", nodata=-3.4028230607370965e+38)
    # Append mean population density to the grid
    grid["pop_density"] = [s["mean"] for s in stats]
    return grid

def add_road_distance_covariate(grid, roads_fp):
    """
    For each grid cell, compute the distance from its centroid to the nearest road.
    Use a projected CRS (EPSG:3857) for accurate distance measurements.
    """
    print("Computing road proximity...")
    # Load roads
    roads = gpd.read_file(roads_fp)
    # Dissolve all road geometries into one MultiLineString (for efficiency)
    roads_union = roads.unary_union
    
    # Reproject grid to EPSG:3857 for distance calculation
    grid_proj = grid.to_crs(epsg=3857)
    
    # Compute centroid distances for each grid cell
    grid_proj["centroid"] = grid_proj.geometry.centroid
    grid_proj["road_dist"] = grid_proj["centroid"].apply(lambda pt: pt.distance(roads_union))
    
    # Merge the road distance back into original grid (in EPSG:4326)
    grid["road_dist"] = grid_proj["road_dist"].values
    return grid

# -----------------------------
# Step 3: Prepare Data for Matching
# -----------------------------
def prepare_data():
    # Load grid with protection indicator and deforestation outcome
    grid = load_grid()
    
    # Add population density covariate (ensure the population raster file exists)
    pop_fp = os.path.join("..", "data", "population", "gpw_v4_popdensity_2020.tif")
    grid = add_population_covariate(grid, pop_fp)
    
    # Add road proximity covariate (distance in meters)
    roads_fp = os.path.join("..", "data", "roads", "roads.geojson")
    grid = add_road_distance_covariate(grid, roads_fp)
    
    # Drop rows with NaN deforestation values (cells outside the mosaic)
    grid = grid.dropna(subset=["lossyear_mean"]).copy()
    
    # Define outcome: we use 'lossyear_mean' as a proxy for deforestation timing/rate.
    outcome = grid["lossyear_mean"]
    
    # Treatment indicator: 'is_protected' (1 = protected, 0 = unprotected)
    y = grid["is_protected"]
    
    # Covariates: Use population density and road distance.
    covariate_columns = ["pop_density", "road_dist"]
    X = grid[covariate_columns]
    
    return X, y, outcome, covariate_columns, grid

# -----------------------------
# Step 4: Estimate Propensity Scores
# -----------------------------
def estimate_propensity(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    propensity = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, propensity)
    print(f"Propensity score model AUC: {auc:.3f}")
    return propensity, model

def plot_propensity_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="propensity_score", hue="is_protected", element="step", stat="density", bins=30)
    plt.title("Propensity Score Distribution by Protection Status")
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.legend(title="Protected", labels=["Unprotected", "Protected"])
    plt.show()

# -----------------------------
# Step 5: Nearest-Neighbor Matching
# -----------------------------
def nearest_neighbor_matching(df, caliper=0.1):
    treated = df[df['is_protected'] == 1].copy()
    control = df[df['is_protected'] == 0].copy()
    matches = []
    for idx, row in treated.iterrows():
        diff = abs(control['propensity_score'] - row['propensity_score'])
        min_diff = diff.min()
        if min_diff <= caliper:
            match_idx = diff.idxmin()
            matches.append((idx, match_idx))
            control = control.drop(match_idx)  # matching without replacement
    matched_rows = []
    for t_idx, c_idx in matches:
        treated_row = df.loc[t_idx].copy()
        control_row = df.loc[c_idx].copy()
        treated_row['matched_id'] = t_idx
        control_row['matched_id'] = t_idx
        matched_rows.append(treated_row)
        matched_rows.append(control_row)
    matched_df = pd.DataFrame(matched_rows)
    print(f"Number of matched pairs: {len(matches)}")
    return matched_df

# -----------------------------
# Step 6: Compute ATT
# -----------------------------
def compute_ATT(matched_df, outcome_col="lossyear_mean"):
    treated_outcomes = matched_df[matched_df['is_protected'] == 1][outcome_col]
    control_outcomes = matched_df[matched_df['is_protected'] == 0][outcome_col]
    att = treated_outcomes.mean() - control_outcomes.mean()
    print(f"Estimated ATT (treated - control): {att:.4f}")
    return att

# -----------------------------
# Main: Run All Steps
# -----------------------------
def main():
    X, y, outcome, covariate_columns, df = prepare_data()
    propensity, model = estimate_propensity(X, y)
    df['propensity_score'] = propensity
    plot_propensity_distribution(df)
    
    matched_df = nearest_neighbor_matching(df, caliper=0.1)
    
    print("Pre-matching covariate means:")
    print(df.groupby('is_protected')[covariate_columns].mean())
    print("Post-matching covariate means:")
    print(matched_df.groupby('is_protected')[covariate_columns].mean())
    
    att = compute_ATT(matched_df, outcome_col="lossyear_mean")
    
    out_fp = os.path.join("..", "data", "processed", "matched_data.csv")
    matched_df.to_csv(out_fp, index=False)
    print("Matched data saved to:", out_fp)

if __name__ == "__main__":
    main()
