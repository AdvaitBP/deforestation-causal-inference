import rasterio
from rasterio.enums import Resampling
import os

def main():
    mosaic_fp = os.path.join("..", "data", "gfc", "lossyear_mosaic.tif")
    downsample_fp = os.path.join("..", "data", "gfc", "lossyear_mosaic_downsample.tif")
    
    with rasterio.open(mosaic_fp) as src:
        scale_factor = 10  # Downsample by factor of 10.
        new_width = src.width // scale_factor
        new_height = src.height // scale_factor
        
        # Read data with resampling.
        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.nearest
        )
        
        # Calculate new transform.
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Update metadata.
        out_meta = src.meta.copy()
        out_meta.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })
        
    # Write the downsampled mosaic to file.
    with rasterio.open(downsample_fp, "w", **out_meta) as dst:
        dst.write(data, 1)
        
    print("Downsampled mosaic saved to:", downsample_fp)

if __name__ == "__main__":
    main()
