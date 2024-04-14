import pandas as pd

url = 'https://sites.ecmwf.int/data/c3sci/bulletin/202402/press_release/PR_fig4_timeseries_era5_sst_daily_60S-60N_1979-2024.csv'
data = pd.read_csv(url)
print(data.head())