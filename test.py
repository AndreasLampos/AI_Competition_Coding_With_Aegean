import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# -------------------------------
# Step 1: Load and inspect the data
# -------------------------------
df = pd.read_csv('deepthink_data.csv')
print("First few rows of the dataset:")
print(df.head())

# -------------------------------
# Step 2: Define features and targets
# -------------------------------
# Now we have separate capacity columns for domestic and international models.
features_D = ['avg_fare_D', 'Selling Prices', 'capacities_D', 'Month_Rank']
features_I = ['avg_fare_I', 'Selling Prices', 'capacities_I', 'Month_Rank']
target_pax_D = 'pax_D'
target_pax_I = 'pax_I'

print("\nSummary statistics of the features and target variables:")
print(df[features_D + features_I + [target_pax_D, target_pax_I]].describe())

# -------------------------------
# Step 3: Split data into training and testing sets for both models
# -------------------------------
X_train_D, X_test_D, y_train_D, y_test_D = train_test_split(
    df[features_D], df[target_pax_D], test_size=0.2, random_state=42
)
X_train_I, X_test_I, y_train_I, y_test_I = train_test_split(
    df[features_I], df[target_pax_I], test_size=0.2, random_state=42
)

print(f"\nNumber of training samples for pax_D: {len(X_train_D)}")
print(f"Number of testing samples for pax_D: {len(X_test_D)}")
print(f"Number of training samples for pax_I: {len(X_train_I)}")
print(f"Number of testing samples for pax_I: {len(X_test_I)}")

# -------------------------------
# Step 4: Standardize the features using separate scalers
# -------------------------------
# For domestic features
scaler_D = StandardScaler()
X_train_D = scaler_D.fit_transform(X_train_D)
X_test_D = scaler_D.transform(X_test_D)

# For international features
scaler_I = StandardScaler()
X_train_I = scaler_I.fit_transform(X_train_I)
X_test_I = scaler_I.transform(X_test_I)

print("\nFirst few rows of the standardized features for pax_D:")
print(X_train_D[:5])
print("\nFirst few rows of the standardized features for pax_I:")
print(X_train_I[:5])

# -------------------------------
# Step 5: Define a function to evaluate models
# -------------------------------
def evaluate_model(y_true, y_pred, model_name, target_name):
    mae = mean_absolute_error(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    print(f'\n{model_name} {target_name} - Mean Absolute Error: {mae:.2f}, Bias: {bias:.2f}, MAE + |Bias|: {mae + abs(bias):.2f}')

# -------------------------------
# Step 6: Train Linear Regression models for both targets
# -------------------------------
# For Domestic Passengers (pax_D)
model_D_lr = LinearRegression()
model_D_lr.fit(X_train_D, y_train_D)
y_pred_D_lr = model_D_lr.predict(X_test_D)
print("\nPredictions for pax_D using Linear Regression:")
print(list(y_pred_D_lr))
print("\nActual values for pax_D:")
print(list(y_test_D))
evaluate_model(y_test_D, y_pred_D_lr, 'Linear Regression', 'pax_D')

# For International Passengers (pax_I)
model_I_lr = LinearRegression()
model_I_lr.fit(X_train_I, y_train_I)
y_pred_I_lr = model_I_lr.predict(X_test_I)
print("\nPredictions for pax_I using Linear Regression:")
print(list(y_pred_I_lr))
print("\nActual values for pax_I:")
print(list(y_test_I))
evaluate_model(y_test_I, y_pred_I_lr, 'Linear Regression', 'pax_I')

# -------------------------------
# Step 7: Interactive User Input for a Month's Features
# -------------------------------
def predict_for_new_month():
    """
    Allows the user to input monthly feature values,
    then predicts both pax_D and pax_I using the trained models.
    """
    print("\n--- New Month Prediction ---")
    try:
        # Get inputs for domestic and international avg fares separately.
        avg_fare_D = float(input("Enter avg_fare_D for domestic (e.g., 27): "))
        avg_fare_I = float(input("Enter avg_fare_I for international (e.g., 72): "))
        selling_prices = float(input("Enter Selling Prices (e.g., 103.13): "))
        capacity_D = float(input("Enter capacity_D for domestic (e.g., 99837.0): "))
        capacity_I = float(input("Enter capacity_I for international (e.g., 110000.0): "))
        month_rank = float(input("Enter Month_Rank (e.g., 10): "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return

    # Create DataFrames with the same column names as used during training.
    new_features_D = pd.DataFrame([[avg_fare_D, selling_prices, capacity_D, month_rank]], columns=features_D)
    new_features_I = pd.DataFrame([[avg_fare_I, selling_prices, capacity_I, month_rank]], columns=features_I)
    
    # Scale the new features using the respective scalers.
    new_features_D_scaled = scaler_D.transform(new_features_D)
    new_features_I_scaled = scaler_I.transform(new_features_I)
    
    # Predict using the trained models.
    pred_pax_D = model_D_lr.predict(new_features_D_scaled)[0]
    pred_pax_I = model_I_lr.predict(new_features_I_scaled)[0]
    
    print("\n--- Prediction Results ---")
    print(f"Predicted Domestic Passengers (pax_D): {pred_pax_D:.2f}")
    print(f"Predicted International Passengers (pax_I): {pred_pax_I:.2f}")

# -------------------------------
# Step 8: Run the interactive prediction
# -------------------------------
predict_for_new_month()
