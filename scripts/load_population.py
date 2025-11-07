import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import os

def main():
    pop_fp = os.path.join("..", "data", "population", "gpw_v4_popdensity_2020.tif")
    with rasterio.open(pop_fp) as src:
        print("Population Density Raster Profile:")
        print(src.profile)
        
        # Downsample by factor of 50 for quick plotting
        scale_factor = 50
        new_height = src.height // scale_factor
        new_width = src.width // scale_factor
        
        # Read with resampling
        pop_data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.nearest
        )
    
    # Plot the downsampled data
    plt.figure(figsize=(8,6))
    plt.imshow(pop_data, cmap="plasma")
    plt.colorbar(label="Population Density")
    plt.title("Population Density (Downsampled)")
    plt.show()

if __name__ == "__main__":
    main()
