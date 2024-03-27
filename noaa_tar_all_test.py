import requests, tarfile, os

# URL of the .tar.gz file
url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz"
local_tar_path = "ghcnd_all.tar.gz"

# Step 1: Download the .tar.gz file
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(local_tar_path, 'wb') as f:
        f.write(response.raw.read())
    print("Downloaded the tar.gz file successfully.")

    # Step 2: Extract the file contents
    # Check if the tar file exists to avoid errors
    if os.path.exists(local_tar_path):
        with tarfile.open(local_tar_path, "r:gz") as tar:
            # This extracts all files. If you know the specific files you need, you can extract them selectively.
            tar.extractall(path="ghcnd_all")
        print("Extracted the tar.gz file successfully.")
else:
    print(f"Failed to download the file, status code: {response.status_code}")
