import rasterio
from rasterio.merge import merge
import os

def main():
    # Define file paths for the three lossyear tiles
    tile_files = [
        os.path.join("..", "data", "gfc", "Hansen_GFC-2023-v1.11_lossyear_10N_070W.tif"),
        os.path.join("..", "data", "gfc", "Hansen_GFC-2023-v1.11_lossyear_00N_070W.tif"),
        os.path.join("..", "data", "gfc", "Hansen_GFC-2023-v1.11_lossyear_00N_080W.tif")
    ]
    
    # Open each tile
    src_files_to_mosaic = []
    for fp in tile_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    
    # Merge the rasters; this creates a mosaic array and an affine transform
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Update the metadata with the mosaic shape, transform, etc.
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })
    
    # Define output file path for mosaic
    out_fp = os.path.join("..", "data", "gfc", "lossyear_mosaic.tif")
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)
    print("Mosaic created and saved to:", out_fp)
    
    # Close all source files
    for src in src_files_to_mosaic:
        src.close()

if __name__ == "__main__":
    main()
