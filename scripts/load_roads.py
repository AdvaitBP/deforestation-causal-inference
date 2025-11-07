import geopandas as gpd
import os

input_path = r"..\data\roads\norte-latest-free\gis_osm_roads_free_1.shp"  # adjust if necessary
roads = gpd.read_file(input_path)

# Inspect available columns (optional)
print("Columns in roads shapefile:", roads.columns.tolist())

# Use the 'fclass' field (most OSM extracts label road types there)
roads = roads[roads["fclass"].notnull()]

# Optionally filter for major road types:
major_types = ["motorway", "trunk", "primary", "secondary", "tertiary"]
roads = roads[roads["fclass"].isin(major_types)]

roads = roads.to_crs("EPSG:4326")
output_path = r"..\data\roads\roads.geojson"
roads.to_file(output_path, driver="GeoJSON")
print(f"Exported {len(roads)} road features to {output_path}")
