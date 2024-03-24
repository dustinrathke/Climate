import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load dataset
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

# Linear Regression
model_linear = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_linear.fit(X_train, y_train)

# Initial Polynomial Regression (Degree 2)
poly_features_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_features_2.fit_transform(X)
model_poly_2 = LinearRegression().fit(X_poly_2, y)

# Refined Polynomial Regression with Cross-Validation and Ridge
degrees = [2, 3, 4, 6, 8]
best_score = float('-inf')
best_model = None

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=1.0))
    scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    if np.mean(scores) > best_score:
        best_score = np.mean(scores)
        best_degree = degree
        best_model = model

best_model.fit(X, y)

# Predicting with Best Model
future_years = np.arange(2020, 2041).reshape(-1, 1)
future_temps_pred = best_model.predict(future_years)

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='blue', label='Actual Temperature Changes')
plt.plot(X, model_linear.predict(X), color='red', label='Linear Regression')
plt.plot(X, model_poly_2.predict(X_poly_2), color='green', label='Polynomial Regression (Degree 2)')
plt.plot(future_years, future_temps_pred, 'k--', label=f'Best Polynomial Regression (Degree {best_degree})')

# Highlighting the year 2030
specific_year = 2030
specific_year_pred = future_temps_pred[future_years.flatten() == specific_year][0]
plt.scatter([specific_year], [specific_year_pred], color='orange', zorder=5)
plt.text(specific_year, specific_year_pred, f'{specific_year_pred:.2f}°C', color='orange', ha='right', va='bottom')

plt.title('Temperature Change Predictions')
plt.xlabel('Year')
plt.ylabel('Temperature Change (°C)')
plt.legend()
plt.grid(True)
plt.show()
