import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta

# Load the dataset
df = pd.read_csv('SanMarcos_Station_Daily_Max_Temp.csv')

# Convert 'DATE' column to datetime format and sort the DataFrame
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)

# Remove rows with NULL 'TMAX' values
df = df.dropna(subset=['TMAX'])

# Decompose the time series to identify and separate the trend component
decomposition = seasonal_decompose(df['TMAX'], model='additive', period=365, extrapolate_trend='freq')

# Isolate the trend component
trend = decomposition.trend.dropna()  # Drops the NaNs which might be present at the ends

# Prepare the trend data for regression analysis
trend_df = pd.DataFrame({'DATE': df['DATE'], 'TMAX_trend': trend})
trend_df.dropna(inplace=True)  # Ensure no NaNs are present
trend_df['date_ordinal'] = trend_df['DATE'].apply(lambda x: x.toordinal())

# Perform regression on the trend
X_trend = trend_df[['date_ordinal']]
y_trend = trend_df['TMAX_trend']
model_trend = LinearRegression()
model_trend.fit(X_trend, y_trend)

# Retrieve the slope (coefficient) and the y-intercept from the regression model
slope = model_trend.coef_[0]
intercept = model_trend.intercept_

# Forecasting
# Create a range of dates from the earliest to the latest date plus one year for future prediction
date_range = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max() + pd.Timedelta(days=365))
date_range_ordinal = date_range.map(pd.Timestamp.toordinal)

# Predict using the date range
trend_pred = model_trend.predict(date_range_ordinal.values.reshape(-1, 1))

# Initial and final temperatures of the trend
initial_trend_temp = trend_pred[0]
final_trend_temp = trend_pred[-1]

# Visualization
plt.figure(figsize=(12, 6))

# Plot the actual TMAX values
plt.scatter(df['DATE'], df['TMAX'], color='lightblue', label='Actual TMAX', alpha=0.5)

# Plot the trend component along with the forecasted values
plt.plot(date_range, trend_pred, label='Trend and Forecasted Trend', color='orange')

# Annotate the initial and final trend temperatures
plt.annotate(f'Initial Trend Temp: {initial_trend_temp:.2f}°', (date_range[0], initial_trend_temp), 
             textcoords="offset points", xytext=(-15,-10), ha='center')
plt.annotate(f'Final Trend Temp: {final_trend_temp:.2f}°', (date_range[-1], final_trend_temp), 
             textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Date')
plt.ylabel('TMAX')
plt.title('Trend and Forecasted Trend for TMAX with Initial and Final Temperatures')
plt.legend()

# Show the equation of the trend line on the plot
plt.text(0.01, 0.99, f'y = {slope:.4f}x + {intercept:.4f}', transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top')

plt.show()
