import os
import requests

def download_file(url, output_path):
    print(f"Downloading: {url}")
    response = requests.get(url)
    response.raise_for_status()  # stop if there's an error
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Saved to: {output_path}")

def main():
    base_url = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/"
    # Define our three tiles: note that "00N_070W" already exists in your data folder.
    tiles = ["10N_070W", "00N_070W", "00N_080W"]
    layers = ["treecover2000", "lossyear", "datamask", "gain"]

    # Data directory (relative path from scripts folder)
    data_dir = os.path.join("..", "data", "gfc")
    os.makedirs(data_dir, exist_ok=True)

    for tile in tiles:
        for layer in layers:
            filename = f"Hansen_GFC-2023-v1.11_{layer}_{tile}.tif"
            url = base_url + filename
            output_path = os.path.join(data_dir, filename)
            if not os.path.exists(output_path):
                try:
                    download_file(url, output_path)
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
            else:
                print(f"File {filename} already exists, skipping.")

if __name__ == "__main__":
    main()
