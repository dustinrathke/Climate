from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Using a simple temperature record dataset from Kaggle we will fit a linear regression model to generate a trend-line
# Then we will refine the fitting of the trend by switching to a polynomial regression
# And finally with a good fit, we will generate a prediction based on the polynomial regression to predict global temperature by 2030

# Load dataset
data = pd.read_csv("C:/Users/Dustin Winter/OneDrive/Data/Climate/climate_change_indicators.csv")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the training set and the test set
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate the performance metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

(train_mse, test_mse, train_r2, test_r2), model.coef_[0], model.intercept_

print(f"Linear Regression Model for Predicting Temperature Changes:")
print(f"-------------------------------------------------------------")
print(f"Coefficient (Temperature Change per Year): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"")
print(f"Performance Metrics:")
print(f"- MSE on Training Data: {train_mse:.4f}")
print(f"- MSE on Testing Data: {test_mse:.4f}")
print(f"- R² Score on Training Data: {train_r2:.4f}")
print(f"- R² Score on Testing Data: {test_r2:.4f}")

# Plotting the historical data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Temperature Changes')

# Plotting the regression line
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Linear Regression Fit Over Temperature Records')

# Adding titles and labels
plt.title('Simple Linear Regression Fit by Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

from sklearn.preprocessing import PolynomialFeatures

# Choose a degree for the polynomial features
poly_degree = 2

# Generate polynomial features
poly_features = PolynomialFeatures(degree=poly_degree)
X_poly = poly_features.fit_transform(X)

# Split the polynomial features into training and testing datasets
X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create and train the model on the polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Predict using the polynomial model
y_poly_pred = poly_model.predict(X_poly)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Temperature Changes')
plt.plot(X, y_poly_pred, color='green', linewidth=2, label='Polynomial Model Predictions')
plt.title('Polynomial Regression Model Fit over Temperature Change')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.legend()
plt.grid(True)
plt.show()


# Let's redo the polynomial regression prediction and plotting, starting from after the 2nd graph, ensuring all variables are defined.

# For clarity, let's redefine our steps and variables from generating polynomial features to making future predictions and plotting.

# 1. Generate polynomial features for the original dataset years.
poly_features = PolynomialFeatures(degree=2)  # Using a 2nd degree polynomial
X_poly = poly_features.fit_transform(X)  # Transforming our original years data into polynomial features

# 2. Train the polynomial regression model (using linear regression on polynomial features).
poly_model = LinearRegression()
poly_model.fit(X_poly, y)  # Training the model on the entire dataset for simplicity

# 3. Predict temperatures for the original dataset years using the trained model.
y_poly_pred = poly_model.predict(X_poly)

# 4. Define future years (2023 to 2030) for prediction and predict future temperatures.
future_years = np.arange(2023, 2041).reshape(-1, 1)  # Future years we're interested in
future_years_poly = poly_features.transform(future_years)  # Transforming future years into polynomial features
future_temps_pred = poly_model.predict(future_years_poly)  # Predicting future temperatures

# 5. Combine the original dataset years and future years for plotting.
all_years = np.vstack((X, future_years))  # Combining original and future years
all_temps_pred = np.concatenate((y_poly_pred, future_temps_pred))  # Combining predictions for a continuous plot

# Plotting the results with historical data and future predictions
# Adding a value label to the last point on the graph for the future predictions

# Plotting the results with historical data and future predictions, including a label for the last future prediction point
plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='blue', label='Actual Historical Temperature Changes')  # Actual data points
plt.plot(all_years, all_temps_pred, color='black', linewidth=2, label='Polynomial Regression Predictions')  # Model predictions
plt.scatter(future_years, future_temps_pred, color='red', label='Future Predictions')  # Highlighting future predictions

specific_year = 2030
specific_year_index = np.where(future_years == specific_year)[0][0]  # Finding the index of the year 2030
specific_temp_pred = future_temps_pred[specific_year_index]
plt.text(specific_year, specific_temp_pred, f'{specific_temp_pred:.2f}°C', color='red', ha='right', va='bottom')

# Adding a label to the last future prediction point
last_year = future_years[-1, 0]
last_temp_pred = future_temps_pred[-1]
plt.text(last_year, last_temp_pred, f'{last_temp_pred:.2f}°C', color='red', ha='right', va='bottom')

plt.title('Polynomial Regression Preditive Analysis: Historical Data & Future Predictions')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.ylim(bottom=min(y)-0.1, top=3.0)  # Adjusting y-axis
plt.legend()
plt.grid(True)
plt.show()