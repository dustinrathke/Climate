import pandas as pd
import requests
import tarfile
from io import StringIO

# Function to fetch and parse station metadata
def fetch_station_metadata(url):
    colspecs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 75), (76, 79), (80, 85)]
    names = ['ID', 'Latitude', 'Longitude', 'Elevation', 'State', 'Name', 'GSN Flag', 'HCN/CRN Flag', 'WMO ID']
    response = requests.get(url)
    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.read_fwf(data, colspecs=colspecs, names=names)
        return df
    else:
        print("Failed to download the station metadata.")
        return pd.DataFrame()

# Function to lookup station IDs by name
def get_station_name_by_name(df, station_name):
    return df[df['Name'].str.contains(station_name, case=False, na=False)]

# Function to extract data for a specific station ID
def extract_station_data(tar_path, station_name, extraction_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        file_name = f"{station_name}.dly"
        try:
            tar.extract(file_name, path=extraction_path)
            print(f"Extracted {file_name} successfully.")
        except KeyError:
            print(f"File {file_name} not found in the archive.")

# Main workflow
if __name__ == "__main__":
    # Step 1: Fetch station metadata
    station_metadata_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    station_metadata_df = fetch_station_metadata(station_metadata_url)

    # Step 2: User input for station name and lookup
    station_name = input("Enter station name: ")
    matched_stations = get_station_name_by_name(station_metadata_df, station_name)
    print(matched_stations)

    # Assuming the user or script selects an ID from `matched_stations`
    if not matched_stations.empty:
        selected_station_id = matched_stations['ID'].iloc[0]  # Example: taking the first match
        
        # Step 3: Download the .tar.gz file (skipped for brevity; see previous snippets for guidance)
        local_tar_path = "ghcnd_all.tar.gz"  # Assume this is already downloaded
        extraction_path = "./data"
        
        # Extract data for the selected station ID
        extract_station_data(local_tar_path, selected_station_id, extraction_path)
    ##elif not m
    
    else:
        print("No matching station found.")

