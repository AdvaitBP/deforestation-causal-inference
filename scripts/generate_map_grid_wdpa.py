# scripts/generate_map_grid_wdpa.py

import geopandas as gpd
import matplotlib.pyplot as plt

# 1) load your grid (with lossyear_mean) and WDPA boundaries
grid = gpd.read_file("data/processed/grid_stats_with_protection.geojson")
wdpa = gpd.read_file("data/processed/wdpa_brazil.geojson")

# 2) restrict WDPA to the grid’s bounding box
minx, miny, maxx, maxy = grid.total_bounds
wdpa_clip = wdpa.cx[minx:maxx, miny:maxy]

# 3) set up the plot
fig, ax = plt.subplots(figsize=(8, 6))

# 4) plot the forest‐loss raster (as colored grid cells)
grid.plot(
    column="lossyear_mean",
    cmap="viridis",
    scheme="quantiles",
    k=5,
    edgecolor="none",
    legend=True,
    legend_kwds={
        "title": "Mean annual forest-loss (years until loss)"
    },
    ax=ax,
)

# 5) overlay the clipped WDPA polygons
wdpa_clip.plot(
    facecolor="none",
    edgecolor="pink",
    linewidth=1,
    ax=ax,
    label="Protected Areas"
)

# 6) cosmetic tweaks
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Spatial distribution of mean annual forest-loss\nwith WDPA boundaries")

# 7) optional gridlines / ticks
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

# 8) save
plt.tight_layout()
plt.savefig("map_grid_wdpa.png", dpi=300)
plt.close(fig)
