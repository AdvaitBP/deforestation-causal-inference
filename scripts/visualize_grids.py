import os
import geopandas as gpd
import matplotlib.pyplot as plt

# Paths — adjust if needed!
GRID_FP        = os.path.join("..", "data", "processed", "grid_stats.geojson")
GRID_PROT_FP   = os.path.join("..", "data", "processed", "grid_stats_with_protection.geojson")

# 1. Load the grids
grid     = gpd.read_file(GRID_FP)
grid_prot = gpd.read_file(GRID_PROT_FP)

# 2. Quick check of CRS
print("Grid CRS:", grid.crs)
print("Grid w/ protection CRS:", grid_prot.crs)

# 3. Plot 1: All cell boundaries, color by protection status
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
grid_prot.plot(
    column="is_protected",
    categorical=True,
    cmap="tab10",
    legend=True,
    legend_kwds={"title": "Protected?"},
    linewidth=0.1,
    edgecolor="gray",
    ax=ax
)
ax.set_title("Grid Cells by Protection Status", fontsize=16)
ax.set_axis_off()

# 4. Plot 2: Choropleth of mean annual loss (lossyear_mean)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
grid.plot(
    column="lossyear_mean",
    cmap="OrRd",
    legend=True,
    legend_kwds={"label": "Mean lossyear value"},
    linewidth=0.1,
    edgecolor="gray",
    ax=ax
)
ax.set_title("Grid Cells by Mean Annual Forest Loss", fontsize=16)
ax.set_axis_off()

plt.show()
