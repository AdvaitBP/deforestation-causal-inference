#!/usr/bin/env python
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1) load your geojson
gdf = gpd.read_file("data/processed/grid_stats_with_protection.geojson")
# 2) print columns once to see what the treatment column _actually_ is
print("GeoDataFrame columns:", gdf.columns.tolist())

# ▶ On first run, note the name here and below (e.g. 'is_protected', 'T', 'PA')
tcol = "is_protected"   # ← REPLACE this with whatever column you saw above

# 3) Histograms for treated vs. control
for val, label in [(1,"treated"), (0,"control")]:
    arr = gdf.loc[gdf[tcol]==val, "lossyear_mean"].dropna()
    plt.figure()
    plt.hist(arr, bins=30)
    plt.title(f"Histogram of mean annual loss (lossyear_mean) — {label}")
    plt.xlabel("lossyear_mean")
    plt.ylabel("Frequency")
    plt.savefig(f"hist_lossyear_{label}.png", dpi=300)
    plt.close()

# 4) Boxplot from your matched_data
df = pd.read_csv("data/processed/matched_data.csv")
print("matched_data columns:", df.columns.tolist())
bcol = "is_protected"   # ← likely the same name; adjust if needed

plt.figure()
df.boxplot(column="lossyear_mean", by=bcol)
plt.title("Boxplot of mean annual forest-loss for matched treated vs control")
plt.suptitle("")   # remove automatic “Boxplot grouped by …”
plt.xlabel("Treatment")
plt.ylabel("lossyear_mean")
plt.savefig("boxplot_lossyear.png", dpi=300)
plt.close()

# 5) Randomization-inference null distribution
#    (You need to have saved your permuted‐ATTs as a NumPy array ri_null_distribution.npy)
try:
    ri = np.load("ri_null_distribution.npy")
    plt.figure()
    plt.hist(ri, bins=30)
    plt.axvline(0.2621, color='k', linestyle="--")  # observed ATT
    plt.title("Randomization-inference null distribution of permuted ATTs")
    plt.xlabel("ATT")
    plt.ylabel("Frequency")
    plt.savefig("ri_distribution.png", dpi=300)
    plt.close()
except FileNotFoundError:
    print("  › ri_null_distribution.npy not found; skipping RI figure.")
