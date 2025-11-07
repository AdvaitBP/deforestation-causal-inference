import os
import osmnx as ox

def main():
    # Define a place name covering your study area.
    # For example, "Amazonas, Brazil" (adjust as needed for your study region)
    place_name = "Amazonas, Brazil"
    print(f"Downloading road network data for {place_name}...")
    
    # Download the road network using graph_from_place
    G = ox.graph_from_place(place_name, network_type="drive")
    roads = ox.graph_to_gdfs(G, nodes=False)
    roads = roads.to_crs("EPSG:4326")
    
    # Save roads data to the data/roads folder
    data_dir = os.path.join("..", "data", "roads")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "roads.geojson")
    roads.to_file(output_path, driver="GeoJSON")
    print(f"Saved road network data to: {output_path}")

if __name__ == "__main__":
    main()
