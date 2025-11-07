import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point

def load_grid():
    # Load the grid with zonal statistics from preprocess_data.py
    grid_fp = os.path.join("..", "data", "processed", "grid_stats.geojson")
    grid = gpd.read_file(grid_fp)
    return grid

def load_wdpa():
    # Load the WDPA polygons filtered to Brazil
    wdpa_fp = os.path.join("..", "data", "processed", "wdpa_brazil.geojson")
    wdpa = gpd.read_file(wdpa_fp)
    return wdpa

def add_protection_indicator(grid, wdpa):
    # Assign a unique identifier to each grid cell
    grid = grid.reset_index(drop=True).copy()
    grid["grid_id"] = grid.index
    
    # Reproject grid and WDPA to a projected CRS for accurate centroid calculation
    grid_proj = grid.to_crs(epsg=3857)
    wdpa_proj = wdpa.to_crs(epsg=3857)
    
    # Compute centroids on the projected grid
    grid_proj["centroid"] = grid_proj.geometry.centroid
    
    # Perform a spatial join using the centroids.
    joined = gpd.sjoin(grid_proj.set_geometry("centroid"), wdpa_proj, how="left", predicate="within")
    
    # Group by grid_id and mark as protected (1) if any match is found, else 0.
    protection = joined.groupby("grid_id")["index_right"].apply(lambda x: 1 if x.notnull().any() else 0)
    
    # Map the protection indicator back to the original grid using grid_id
    grid["is_protected"] = grid["grid_id"].map(protection).fillna(0).astype(int)
    
    return grid

def plot_deforestation_distribution(df):
    # Plot histogram/density of deforestation rates (lossyear_mean)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="lossyear_mean", hue="is_protected", element="step", stat="density", bins=30)
    plt.title("Distribution of Deforestation Rates (Protected vs. Unprotected)")
    plt.xlabel("Mean Lossyear Value (proxy for deforestation rate)")
    plt.ylabel("Density")
    plt.legend(title="Protected", labels=["Unprotected", "Protected"])
    plt.show()
    
    # Print summary statistics by protection status
    summary = df.groupby("is_protected")["lossyear_mean"].describe()
    print("Summary statistics by protection status:")
    print(summary)

def plot_boxplot(df, outcome_col="lossyear_mean"):
    # Plot a boxplot to compare outcomes between groups
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="is_protected", y=outcome_col, data=df)
    plt.title("Boxplot of Deforestation Outcome by Protection Status")
    plt.xlabel("Protection Status (0 = Unprotected, 1 = Protected)")
    plt.ylabel(outcome_col)
    plt.show()

def spatial_plot(grid, wdpa):
    # Plot grid cells colored by deforestation rate and overlay WDPA boundaries
    fig, ax = plt.subplots(figsize=(12, 10))
    grid.plot(column="lossyear_mean", cmap="OrRd", legend=True, ax=ax, alpha=0.7, edgecolor="grey")
    wdpa.boundary.plot(ax=ax, color="blue", linewidth=1, label="WDPA")
    plt.title("Spatial Distribution of Deforestation Rates with WDPA Boundaries")
    plt.legend()
    plt.show()

def main():
    # Load grid and WDPA data
    grid = load_grid()
    wdpa = load_wdpa()
    
    print("Before adding protection indicator:")
    print(grid.head())
    
    # Add protection indicator to grid cells
    grid = add_protection_indicator(grid, wdpa)
    
    print("After adding protection indicator (first 5 rows):")
    print(grid[["lossyear_mean", "is_protected"]].head())
    
    # Drop grid cells with NaN deforestation values (cells outside valid mosaic area)
    grid_clean = grid.dropna(subset=["lossyear_mean"]).copy()
    
    # Plot the deforestation distribution (histogram)
    plot_deforestation_distribution(grid_clean)
    
    # Plot the boxplot for deforestation outcome
    plot_boxplot(grid_clean, outcome_col="lossyear_mean")
    
    # Spatial plot of deforestation rates with WDPA overlay
    spatial_plot(grid_clean, wdpa)
    
    # Save the grid with protection indicator for future use
    out_fp = os.path.join("..", "data", "processed", "grid_stats_with_protection.geojson")
    grid_clean.to_file(out_fp, driver="GeoJSON")
    print("Grid with protection indicator saved to:", out_fp)

if __name__ == "__main__":
    main()
