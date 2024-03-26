import pandas as pd
import requests

# Step 1: Fetch the data from the URL
url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
response = requests.get(url)

# Ensure the request was successful
if response.status_code == 200:
    # Step 2: Read the data into a pandas DataFrame
    # The data is fixed-width, so we use read_fwf instead of read_csv
    data = pd.read_fwf(url, colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 75), (76, 79), (80, 85)], 
                       names=['ID', 'Latitude', 'Longitude', 'Elevation', 'State', 'Name', 'GSN Flag', 'HCN/CRN Flag', 'WMO ID'])

    # Example processing: Print the first few rows to check the data
    print(data.head())

    # More processing can be done here, e.g., filtering, analysis, etc.
else:
    print(f"Failed to fetch data, status code: {response.status_code}")
