from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Placeholder for actual data loading
data = pd.read_csv('climate_change_indicators.csv')

# Calculate the average temperature change per year across all countries
year_columns = [col for col in data.columns if col.startswith('F')]
yearly_averages = data[year_columns].mean()

# Convert yearly averages to a DataFrame
df_yearly_averages = pd.DataFrame(yearly_averages, columns=['TempChange']).reset_index()
df_yearly_averages['Year'] = df_yearly_averages['index'].str[1:].astype(int)  # Extract year from column names
df_yearly_averages.drop(columns=['index'], inplace=True)

# Splitting the data into training and testing datasets
X = df_yearly_averages[['Year']]
y = df_yearly_averages['TempChange']

# Polynomial degrees to consider
degrees = [2]

best_degree = None
best_score = float('-inf')
best_model = None

kf = KFold(n_splits=5, shuffle=True, random_state=33)

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=0.5))
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    avg_score = np.mean(scores)
    
    if avg_score > best_score:
        best_score = avg_score
        best_degree = degree
        best_model = model

# Fit the best model to the entire dataset
best_model.fit(X, y)

# Predict using the best model
y_pred = best_model.predict(X)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Polynomial Fit')
plt.xlabel('Year')
plt.ylabel('Temperature Change')
plt.title('Temperature Change Over Years')

# Set the x-axis range
start_year = X['Year'].min()
end_year = 2040
plt.xlim(start_year, end_year)

plt.legend()
plt.show()

# 3. Predict temperatures for the original dataset years using the trained model.
y_poly_pred = best_model.predict(X)

# 4. Define future years (2023 to 2040) for prediction and predict future temperatures.
future_years = np.arange(2023, 2041).reshape(-1, 1)  # Future years we're interested in
future_temps_pred = best_model.predict(future_years)  # Predicting future temperatures

# 5. Combine the original dataset years and future years for plotting.
all_years = np.vstack((X, future_years))  # Combining original and future years
all_temps_pred = np.concatenate((y_poly_pred, future_temps_pred))  # Combining predictions for a continuous plot

# Printing future temperature predictions
print(f"Optimal Polynomial Degree: {best_degree}")
print("Future Temperature Predictions:")
for year, temp_change in zip(future_years.flatten(), future_temps_pred):
    print(f"Year {year}: {temp_change:.2f}°C")

# Plotting future temperature predictions
plt.figure(figsize=(10, 6))
plt.plot(future_years, future_temps_pred, marker='o', linestyle='-', color='b')
plt.title('Future Temperature Predictions')
plt.xlabel('Year')
plt.ylabel('Predicted Temperature Change (°C)')
plt.grid(True)
plt.xticks(np.arange(start_year, end_year + 1, 5), rotation=45)

# Set the x-axis range for future predictions
plt.xlim(start_year, end_year)

plt.tight_layout()

# Display the plot
plt.show()