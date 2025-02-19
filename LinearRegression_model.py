import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Read the CSV file
df = pd.read_csv('deepthink_data.csv')

# Define features and targets
X = df[['avg_fare_D', 'avg_fare_I', 'capacities_D', 'capacities_I', 'month_rank']]
y_D = df['pax_D']
y_I = df['pax_I']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_D_train, y_D_test, y_I_train, y_I_test = train_test_split(
    X, y_D, y_I, test_size=0.2, random_state=42
)

# Create and train models for domestic passengers
model_D = LinearRegression()
model_D.fit(X_train, y_D_train)

# Create and train models for international passengers
model_I = LinearRegression()
model_I.fit(X_train, y_I_train)

# Make predictions
y_D_pred = model_D.predict(X_test)
y_I_pred = model_I.predict(X_test)

# Calculate accuracy metrics
r2_D = r2_score(y_D_test, y_D_pred)
rmse_D = np.sqrt(mean_squared_error(y_D_test, y_D_pred))

r2_I = r2_score(y_I_test, y_I_pred)
rmse_I = np.sqrt(mean_squared_error(y_I_test, y_I_pred))

print("\nDomestic Passengers (pax_D) Model Performance:")
print(f"R² Score: {r2_D:.4f}")
print(f"RMSE: {rmse_D:.2f} passengers")

print("\nInternational Passengers (pax_I) Model Performance:")
print(f"R² Score: {r2_I:.4f}")
print(f"RMSE: {rmse_I:.2f} passengers")

# Add example prediction
print("\nExample prediction for first test point:")
example_pred_D = model_D.predict(X_test.iloc[[0]])
example_pred_I = model_I.predict(X_test.iloc[[0]])
print(f"Predicted Domestic Passengers: {example_pred_D[0]:.0f}")
print(f"Actual Domestic Passengers: {y_D_test.iloc[0]:.0f}")
print(f"Predicted International Passengers: {example_pred_I[0]:.0f}")
print(f"Actual International Passengers: {y_I_test.iloc[0]:.0f}")