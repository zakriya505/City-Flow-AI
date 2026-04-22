# src/ingestion/data_downloader.py
import requests
import os

# NYC TLC Open Data - Yellow Taxi Trip Records
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
MONTHS = ["2023-01", "2023-02", "2023-03", "2023-04",
          "2023-05", "2023-06", "2023-07", "2023-08",
          "2023-09", "2023-10", "2023-11", "2023-12"]

def download_tlc_data(output_dir: str = "data/raw/taxi_rides"):
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"

    for month in MONTHS:
        filename = f"yellow_tripdata_{month}.parquet"
        url = f"{base_url}/{filename}"
        out_path = os.path.join(output_dir, filename)

        if not os.path.exists(out_path):
            print(f"Downloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  ✓ Saved {filename}")
            except Exception as e:
                print(f"  ✗ Error downloading {filename}: {e}")
        else:
            print(f"  - {filename} already exists. Skipping.")

if __name__ == "__main__":
    download_tlc_data()
