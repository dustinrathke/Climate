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
        best_degree = degree
        best_score = avg_score
        best_model = model

# Retraining on the full dataset
best_model.fit(X, y)

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
plt.xticks(all_years.flatten(), rotation=45)

plt.tight_layout()

# Display the plot
plt.show()