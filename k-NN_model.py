import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from scipy.optimize import minimize_scalar

# ===============================
# Step 1: Load and Inspect the Data
# ===============================
df = pd.read_csv('deepthink_data.csv')
print("="*50)
print("DATA OVERVIEW")
print("="*50)
print("First few rows of the dataset:")
print(df.head())

# Calculate the average airplane capacity for each month for domestic flights
df['average_airplane_capacity_D'] = df['seats_D'] / df['flights_D']

# Calculate the average airplane capacity for each month for international flights
df['average_airplane_capacity_I'] = df['seats_I'] / df['flights_I']

# Calculate the overall average airplane capacity for domestic flights
average_airplane_capacity_D = df['average_airplane_capacity_D'].mean()

# Calculate the overall average airplane capacity for international flights
average_airplane_capacity_I = df['average_airplane_capacity_I'].mean()

# Print the results
print("\nAverage Airplane Capacity for Each Month for Domestic Flights:")
print(f"\nOverall Average Airplane Capacity (Domestic): {average_airplane_capacity_D:.2f}")

print("\nAverage Airplane Capacity for Each Month for International Flights:")
print(f"Overall Average Airplane Capacity (International): {average_airplane_capacity_I:.2f}")


# ===============================
# Step 2: Define Features and Targets
# ===============================
features_D = ['avg_fare_D', 'selling_prices', 'capacities_D', 'month_rank']
features_I = ['avg_fare_I', 'selling_prices', 'capacities_I', 'month_rank']
target_pax_D = 'pax_D'
target_pax_I = 'pax_I'

print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(df[features_D + features_I + [target_pax_D, target_pax_I]].describe())

# ===============================
# Step 3: Train/Test Split
# ===============================
X_train_D, X_test_D, y_train_D, y_test_D = train_test_split(
    df[features_D], df[target_pax_D], test_size=0.1, random_state=42
)
X_train_I, X_test_I, y_train_I, y_test_I = train_test_split(
    df[features_I], df[target_pax_I], test_size=0.1, random_state=42
)

print("\n" + "="*50)
print("TRAIN/TEST SPLIT")
print("="*50)
print(f"Domestic Training Samples: {len(X_train_D)}")
print(f"Domestic Testing Samples: {len(X_test_D)}")
print(f"International Training Samples: {len(X_train_I)}")
print(f"International Testing Samples: {len(X_test_I)}")

# ===============================
# Step 4: Standardize Features
# ===============================
scaler_D = StandardScaler()
X_train_D = scaler_D.fit_transform(X_train_D)
X_test_D = scaler_D.transform(X_test_D)

scaler_I = StandardScaler()
X_train_I = scaler_I.fit_transform(X_train_I)
X_test_I = scaler_I.transform(X_test_I)

print("\n" + "="*50)
print("STANDARDIZED FEATURES (First 5 Rows)")
print("="*50)
print("Domestic:")
for row in X_train_D[:5]:
    print(row)
print("\nInternational:")
for row in X_train_I[:5]:
    print(row)

# ===============================
# Step 5: Train kNN Regression Models
# ===============================
n_neighbors = 5
model_D_knn = KNeighborsRegressor(n_neighbors=n_neighbors)
model_D_knn.fit(X_train_D, y_train_D)
y_pred_D_knn = model_D_knn.predict(X_test_D)

model_I_knn = KNeighborsRegressor(n_neighbors=n_neighbors)
model_I_knn.fit(X_train_I, y_train_I)
y_pred_I_knn = model_I_knn.predict(X_test_I)

print("\n" + "="*50)
print("PREDICTIONS (kNN Regression)")
print("="*50)
print("Domestic Predictions:")
print([float(x) for x in y_pred_D_knn])
print("Domestic Actuals:")
print(y_test_D.tolist())

print("\nInternational Predictions:")
print([float(x) for x in y_pred_I_knn])
print("International Actuals:")
print(y_test_I.tolist())

# ===============================
# Step 6: Model Evaluation using Cross-Validation
# ===============================
cv_folds = 8    # Number of cross-validation folds for evaluation.

print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)
cv_scores_D = cross_val_score(model_D_knn, scaler_D.transform(df[features_D]), df[target_pax_D], cv=cv_folds, scoring='r2')
cv_scores_I = cross_val_score(model_I_knn, scaler_I.transform(df[features_I]), df[target_pax_I], cv=cv_folds, scoring='r2')

print("Domestic R² Scores per Fold:", cv_scores_D)
print(f"Domestic Average R² Score: {cv_scores_D.mean():.2f}")
print("\nInternational R² Scores per Fold:", cv_scores_I)
print(f"International Average R² Score: {cv_scores_I.mean():.2f}")

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
cv_mae_D = cross_val_score(model_D_knn, scaler_D.transform(df[features_D]), df[target_pax_D], cv=cv_folds, scoring=mae_scorer)
cv_mae_I = cross_val_score(model_I_knn, scaler_I.transform(df[features_I]), df[target_pax_I], cv=cv_folds, scoring=mae_scorer)

print("\nDomestic MAE per Fold (in absolute value):", [float (-score) for score in cv_mae_D])
print(f"Domestic Average MAE: {-cv_mae_D.mean():.2f}")
print("\nInternational MAE per Fold (in absolute value):", [float (-score) for score in cv_mae_I])
print(f"International Average MAE: {-cv_mae_I.mean():.2f}")

train_r2_D = model_D_knn.score(X_train_D, y_train_D)
test_r2_D = model_D_knn.score(X_test_D, y_test_D)
train_r2_I = model_I_knn.score(X_train_I, y_train_I)
test_r2_I = model_I_knn.score(X_test_I, y_test_I)

print("\nDomestic Model:")
print(f"  Training R² Score: {train_r2_D:.2f}")
print(f"  Testing R² Score: {test_r2_D:.2f}")
print("\nInternational Model:")
print(f"  Training R² Score: {train_r2_I:.2f}")
print(f"  Testing R² Score: {test_r2_I:.2f}")

# ===============================
# Step 7: Interactive New Month Prediction & Optimization
# ===============================

def optimize_domestic_fare():
    """
    This function finds the optimal domestic fare such that the predicted demand
    is as close as possible to the provided total capacity while ensuring that
    the fare remains realistic relative to the competitor's selling price.
    
    If the chosen domestic fare exceeds the competitor's selling price, the predicted 
    pax is scaled down proportionally, reflecting that a much higher fare would 
    result in lower demand. The effective load factor is computed based on the 
    adjusted predicted pax.
    
    In addition, using the overall average airplane capacity (average_airplane_capacity_D),
    we calculate how many flights are needed and the load factor per flight.
    """
    print("\n" + "="*50)
    print("OPTIMIZATION OF DOMESTIC FARE")
    print("="*50)
    try:
        init_avg_fare_D = float(input("Enter initial avg_fare_D for domestic (e.g., 27): "))
        selling_prices = float(input("Enter selling_prices (e.g., 103.13): "))
        total_capacity = float(input("Enter total capacity for domestic (e.g., 99837.0): "))
        month_rank = float(input("Enter month_rank (e.g., 10): "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return
    
    # Define the objective function: minimize (adjusted predicted pax - total_capacity)^2.
    def objective(avg_fare):
        new_features = pd.DataFrame({
            'avg_fare_D': [avg_fare],
            'selling_prices': [selling_prices],
            'capacities_D': [total_capacity],
            'month_rank': [month_rank]
        })
        new_features_scaled = scaler_D.transform(new_features)
        predicted = model_D_knn.predict(new_features_scaled)[0]
        # If the domestic fare is above the competitor's selling price,
        # scale down the predicted pax proportionally.
        adjustment_factor = selling_prices / avg_fare if avg_fare > selling_prices else 1
        adjusted_predicted = predicted * adjustment_factor
        return (adjusted_predicted - total_capacity) ** 2
    
    # Adjust the bounds based on month rank
    lower_bound = 10
    upper_bound = selling_prices * (1 + (12 - month_rank) / 12)  # Adjust upper bound based on month rank
    upper_bound = min(upper_bound, selling_prices * 1.2)  # Ensure the fare does not exceed 120% of selling_prices
    
    # Run the optimizer using the bounded method.
    res = minimize_scalar(objective, bounds=(lower_bound, upper_bound), method='bounded')
    optimal_fare = res.x
    
    # Get the corresponding predicted pax using the optimal fare.
    new_features = pd.DataFrame({
        'avg_fare_D': [optimal_fare],
        'selling_prices': [selling_prices],
        'capacities_D': [total_capacity],
        'month_rank': [month_rank]
    })
    new_features_scaled = scaler_D.transform(new_features)
    predicted = model_D_knn.predict(new_features_scaled)[0]
    adjustment_factor = selling_prices / optimal_fare if optimal_fare > selling_prices else 1
    optimal_predicted = predicted * adjustment_factor
    
    # Compute the effective load factor (capped at 100%).
    effective_LF = min(optimal_predicted, total_capacity) / total_capacity
    
    # Compute the number of flights needed based on the overall average airplane capacity.
    flights_needed = np.ceil(optimal_predicted / average_airplane_capacity_D)
    # Calculate the average load factor per flight.
    load_factor_per_flight = (optimal_predicted / flights_needed) / average_airplane_capacity_D
    
    print("\nFINAL OPTIMIZATION RESULTS (Domestic):")
    print(f"Optimal Average Fare for Domestic (adjusted): {optimal_fare:.2f}")
    print(f"Optimal Predicted Pax (adjusted): {optimal_predicted:.0f}")
    print(f"Provided Total Capacity: {total_capacity:.0f}")
    print(f"Effective Load Factor (with respect to the provided capacity): {effective_LF*100:.2f}%")
    print(f"Overall Average Airplane Capacity (Domestic): {average_airplane_capacity_D:.2f}")
    print(f"Estimated Number of Flights Needed: {int(flights_needed)}")
    print(f"Average Load Factor per Flight: {load_factor_per_flight*100:.2f}%")

def interactive_prediction():
    print("\n" + "="*50)
    print("INTERACTIVE NEW MONTH PREDICTION")
    print("="*50)
    try:
        avg_fare_D = float(input("Enter the Average Fare for Domestic (e.g. 27): "))
        avg_fare_I = float(input("Enter the Average Fare for International (e.g. 72): "))
        selling_prices = float(input("Enter Selling Prices e.g. 103.13): "))
        capacities_D = float(input("Enter Capacities for Domestic (e.g. 99837.0): "))
        capacities_I = float(input("Enter Capacities for International (e.g. 110000.0): "))
        month_rank = float(input("Enter the Month's Rank (e.g. 10): "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return

    new_features_D = pd.DataFrame([[avg_fare_D, selling_prices, capacities_D, month_rank]], columns=features_D)
    new_features_I = pd.DataFrame([[avg_fare_I, selling_prices, capacities_I, month_rank]], columns=features_I)
    
    new_features_D_scaled = scaler_D.transform(new_features_D)
    new_features_I_scaled = scaler_I.transform(new_features_I)
    
    pred_pax_D = model_D_knn.predict(new_features_D_scaled)[0]
    pred_pax_I = model_I_knn.predict(new_features_I_scaled)[0]
    
    print("\n--- Prediction Results ---")
    print(f"Predicted Domestic Passengers (pax_D): {pred_pax_D:.0f}")
    print(f"Predicted International Passengers (pax_I): {pred_pax_I:.0f}")

# ===============================
# Step 8: Run Interactive Components
# ===============================
interactive_prediction()
optimize_domestic_fare()
# ===============================