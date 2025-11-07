import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterstats import zonal_stats
import matplotlib.pyplot as plt

def main():
    # Define the overall bounding box for the study area.
    # Our mosaic covers North=10, South=-10, West=-80, East=-60.
    north, south, west, east = 10, -10, -80, -60
    print(f"Study area bounding box: North {north}, South {south}, West {west}, East {east}")

    # Create a grid over the study area (grid cell size in degrees; adjust as needed)
    grid_size = 0.5
    x_coords = np.arange(west, east, grid_size)
    y_coords = np.arange(south, north, grid_size)
    grid_cells = [box(x, y, x + grid_size, y + grid_size) for x in x_coords for y in y_coords]
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")
    print("Number of grid cells:", len(grid))

    # Use the downsampled mosaic file for zonal statistics.
    lossyear_path = os.path.join("..", "data", "gfc", "lossyear_mosaic_downsample.tif")
    print("Using mosaic file:", lossyear_path)
    stats = zonal_stats(grid, lossyear_path, stats="mean", nodata=0)
    grid["lossyear_mean"] = [s["mean"] for s in stats]
    print("Sample zonal stats (first 5 grid cells):")
    print(grid.head())

    # Save the grid with zonal statistics to the processed folder.
    out_dir = os.path.join("..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "grid_stats.geojson")
    grid.to_file(output_path, driver="GeoJSON")
    print(f"Grid with zonal stats saved to: {output_path}")

    # Optional: Plot the grid colored by mean lossyear value.
    plt.figure(figsize=(10, 8))
    grid.plot(column="lossyear_mean", cmap="OrRd", legend=True)
    plt.title("Mean Lossyear Value per Grid Cell")
    plt.show()

if __name__ == "__main__":
    main()
