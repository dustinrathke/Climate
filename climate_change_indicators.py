from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('climate_change_indicators.csv')

# Using a simple temperature record dataset from Kaggle we will fit a linear regression model to generate a trend-line
# Then we will refine the fitting of the trend by switching to a polynomial regression
# And finally with a good fit, we will generate a prediction based on the polynomial regression to predict global temperature by 2030


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
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Predicted Temperature Changes')

# Adding titles and labels
plt.title('Predicted vs Actual Temperature Changes Over Years')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()



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
plt.title('Polynomial Regression Model vs Actual Temperature Changes')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.legend()
plt.grid(True)
plt.show()






# Feature engineering and scaling within a pipeline
degree = 2  # Consider testing different degrees with cross-validation
pipeline = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    StandardScaler(),
    Ridge(alpha=2.0)  # Regularization strength; can be optimized via cross-validation
)

X = df_yearly_averages[['Year']].values
y = df_yearly_averages['TempChange'].values

# Splitting the data into training and testing datasets for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation to choose the best model
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE: {-np.mean(scores)}")

# Fitting the model
pipeline.fit(X_train, y_train)

# Generating predictions across the range of years and future years for a smoother curve
all_years = np.arange(df_yearly_averages['Year'].min(), 2031).reshape(-1, 1)
predictions = pipeline.predict(all_years)

# Future predictions (2023 to 2030)
future_years = np.arange(2023, 2031).reshape(-1, 1)
future_temps_pred = pipeline.predict(future_years)

# Plotting the results with historical data and future predictions, including a label for the last future prediction point
plt.figure(figsize=(12, 12))
plt.scatter(X, y, color='blue', label='Actual Historical Temperature Changes')  # Actual data points
plt.plot(all_years, predictions, color='green', linewidth=2, label='Polynomial Regression Predictions')  # Model predictions
plt.scatter(future_years, future_temps_pred, color='red', label='Future Predictions')  # Highlighting future predictions

# Adding a label to the last future prediction point
last_year = future_years[-1, 0]
last_temp_pred = future_temps_pred[-1]
plt.text(last_year, last_temp_pred, f'{last_temp_pred:.2f}°C', color='red', ha='left', va='bottom')

plt.title('Temperature Changes: Historical Data & Future Predictions')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.ylim(bottom=min(y)-0.1, top=3.0)  # Adjusting y-axis
plt.legend()
plt.grid(True)
plt.show()
X = data[['Year']]
y = data['TempChange']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with PolynomialFeatures, StandardScaler, and Ridge regression
degree = 2  # Starting point, consider optimizing this based on cross-validation results
pipeline = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                         StandardScaler(),
                         Ridge(alpha=1.0))  # Regularization strength, also a candidate for optimization

# Fitting the model
pipeline.fit(X_train, y_train)

# Making predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.4f}, Testing MSE: {test_mse:.4f}")
print(f"Training R^2: {train_r2:.4f}, Testing R^2: {test_r2:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Temperature Changes')
plt.plot(X, pipeline.predict(X), color='red', label='Model Predictions')
plt.title('Temperature Changes: Model Fit')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.legend()
plt.grid(True)
plt.show()



# Plotting future predictions
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Historical Temperature Changes')
plt.plot(future_years, future_temps_pred, color='red', label='Future Predictions')
plt.title('Temperature Changes: Historical Data & Future Predictions')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.legend()
plt.grid(True)
plt.show()
