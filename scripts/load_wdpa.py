import os
import glob
import pandas as pd
import geopandas as gpd

def merge_shapefiles(shp_dirs, pattern):
    """
    Given a list of directories and a pattern (e.g., "*polygons.shp"),
    load and merge all shapefiles matching the pattern.
    Returns a merged GeoDataFrame.
    """
    shp_files = []
    for folder in shp_dirs:
        files = glob.glob(os.path.join(folder, pattern))
        shp_files.extend(files)
    
    if not shp_files:
        print("No shapefiles found for pattern:", pattern)
        return None
    
    print("Found the following shapefiles:")
    for f in shp_files:
        print(f)
    
    # Load each shapefile and store in a list
    gdf_list = []
    for shp in shp_files:
        gdf = gpd.read_file(shp)
        gdf_list.append(gdf)
    
    # Merge the GeoDataFrames
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
    return merged_gdf

def main():
    # Define the base directory for WDPA data
    wdpa_base_dir = os.path.join("..", "data", "wdpa")
    
    # Folders containing the split shapefiles (polygons)
    polygon_folders = [
        os.path.join(wdpa_base_dir, "WDPA_Mar2025_Public_shp_0"),
        os.path.join(wdpa_base_dir, "WDPA_Mar2025_Public_shp_1"),
        os.path.join(wdpa_base_dir, "WDPA_Mar2025_Public_shp_2")
    ]
    
    # Merge all polygon shapefiles (they all have filenames ending in 'polygons.shp')
    print("Merging WDPA polygon shapefiles from split directories...")
    merged_polygons = merge_shapefiles(polygon_folders, "*polygons.shp")
    if merged_polygons is None:
        return
    print("Total number of polygon records merged:", len(merged_polygons))
    
    # Optional: Merge point shapefiles if needed
    # point_folders = polygon_folders  # typically same folders
    # merged_points = merge_shapefiles(point_folders, "*points.shp")
    # if merged_points is not None:
    #     print("Total number of point records merged:", len(merged_points))
    
    # Filter merged polygons to include only records for Brazil.
    # The filtering field may be "ISO3" or "COUNTRY". Adjust as necessary.
    if "ISO3" in merged_polygons.columns:
        wdpa_brazil = merged_polygons[merged_polygons["ISO3"] == "BRA"].copy()
    else:
        wdpa_brazil = merged_polygons[merged_polygons["COUNTRY"].str.contains("Brazil", case=False, na=False)].copy()
    
    print("Number of protected areas in Brazil:", len(wdpa_brazil))
    
    # Ensure the data are in WGS84 (EPSG:4326)
    wdpa_brazil = wdpa_brazil.to_crs("EPSG:4326")
    
    # Save the merged and filtered WDPA data to the processed folder
    out_dir = os.path.join("..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "wdpa_brazil.geojson")
    wdpa_brazil.to_file(output_path, driver="GeoJSON")
    print(f"Filtered WDPA data saved to: {output_path}")

if __name__ == "__main__":
    main()
