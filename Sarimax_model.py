import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file
df = pd.read_csv('Data Files/deepthink_data.csv')

# Convert month names to numeric values
df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

# Create a datetime index using the 'year' and 'month' columns (assuming day=1 for all)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df.set_index('date', inplace=True)
df.index.freq = 'MS'  # Set the frequency to 'MS' (Month Start)
df.sort_index(inplace=True)

# Define target columns and exogenous features (same as in your XGBoost code)
exog_cols = ['avg_fare_D', 'avg_fare_I', 'capacities_D', 'capacities_I', 'month_rank']
target_D = 'pax_D'
target_I = 'pax_I'

# Split the data into training and testing sets (80% train, 20% test)
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

# Prepare exogenous variables for both train and test
train_exog = train[exog_cols]
test_exog = test[exog_cols]

# Set SARIMA parameters (these may need tuning for your specific dataset)
order = (1, 1, 1)
seasonal_order = (0, 1, 1, 12)  # seasonal period of 12 for monthly data

# ---------------------------
# SARIMA Model for Domestic Passengers (pax_D)
# ---------------------------
model_D = SARIMAX(train[target_D], exog=train_exog, order=order, seasonal_order=seasonal_order)
results_D = model_D.fit(disp=False)
forecast_D = results_D.predict(start=test.index[0], end=test.index[-1], exog=test_exog)

# Evaluate the domestic model
rmse_D = np.sqrt(mean_squared_error(test[target_D], forecast_D))
r2_D = r2_score(test[target_D], forecast_D)

print("\nDomestic Passengers (pax_D) SARIMA Model Performance:")
print(f"R² Score: {r2_D:.4f}")
print(f"RMSE: {rmse_D:.2f} passengers")

# ---------------------------
# SARIMA Model for International Passengers (pax_I)
# ---------------------------
model_I = SARIMAX(train[target_I], exog=train_exog, order=order, seasonal_order=seasonal_order)
results_I = model_I.fit(disp=False)
forecast_I = results_I.predict(start=test.index[0], end=test.index[-1], exog=test_exog)

# Evaluate the international model
rmse_I = np.sqrt(mean_squared_error(test[target_I], forecast_I))
r2_I = r2_score(test[target_I], forecast_I)

print("\nInternational Passengers (pax_I) SARIMA Model Performance:")
print(f"R² Score: {r2_I:.4f}")
print(f"RMSE: {rmse_I:.2f} passengers")

# ---------------------------
# Plot the forecasts against actual data for Domestic Passengers
# ---------------------------
plt.figure(figsize=(14,6))
plt.plot(train.index, train[target_D], label='Train')
plt.plot(test.index, test[target_D], label='Test')
plt.plot(test.index, forecast_D, label='Forecast', color='red')
plt.title("SARIMA Forecast for Domestic Passengers (pax_D)")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()

# ---------------------------
# Plot the forecasts against actual data for International Passengers
# ---------------------------
plt.figure(figsize=(14,6))
plt.plot(train.index, train[target_I], label='Train')
plt.plot(test.index, test[target_I], label='Test')
plt.plot(test.index, forecast_I, label='Forecast', color='red')
plt.title("SARIMA Forecast for International Passengers (pax_I)")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()

# ---------------------------
# Example prediction for the first test point
# ---------------------------
print("\nExample prediction for the first test point:")
example_forecast_D = forecast_D.iloc[0]
example_actual_D = test[target_D].iloc[0]
example_forecast_I = forecast_I.iloc[0]
example_actual_I = test[target_I].iloc[0]

print(f"Predicted Domestic Passengers: {example_forecast_D:.0f}")
print(f"Actual Domestic Passengers: {example_actual_D:.0f}")
print(f"Predicted International Passengers: {example_forecast_I:.0f}")
print(f"Actual International Passengers: {example_actual_I:.0f}")