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

# ===============================
# Step 2: Define Features and Targets
# ===============================
features_D = ['avg_fare_D', 'Selling Prices', 'capacities_D', 'Month_Rank']
features_I = ['avg_fare_I', 'Selling Prices', 'capacities_I', 'Month_Rank']
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
    df[features_D], df[target_pax_D], test_size=0.2, random_state=42
)
X_train_I, X_test_I, y_train_I, y_test_I = train_test_split(
    df[features_I], df[target_pax_I], test_size=0.2, random_state=42
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
cv_folds = 5

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

print("\nDomestic MAE per Fold (in absolute value):", [-score for score in cv_mae_D])
print(f"Domestic Average MAE: {-cv_mae_D.mean():.2f}")
print("\nInternational MAE per Fold (in absolute value):", [-score for score in cv_mae_I])
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
    is as close as possible to the available capacity. Since the model may predict
    demand higher than capacity, we define the effective load factor as:
        Effective LF = min(predicted pax, capacity) / capacity
    The objective function minimizes the squared difference between predicted pax and capacity.
    """
    print("\n" + "="*50)
    print("OPTIMIZATION OF DOMESTIC FARE")
    print("="*50)
    try:
        init_avg_fare_D = float(input("Enter initial avg_fare_D for domestic (e.g., 27): "))
        selling_prices = float(input("Enter Selling Prices (e.g., 103.13): "))
        capacities_D = float(input("Enter capacities_D for domestic (e.g., 99837.0): "))
        month_rank = float(input("Enter Month_Rank (e.g., 10): "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return
    
    # Define the objective function: minimize (predicted pax - capacity)^2.
    def objective(avg_fare):
        new_features = pd.DataFrame({
            'avg_fare_D': [avg_fare],
            'Selling Prices': [selling_prices],
            'capacities_D': [capacities_D],
            'Month_Rank': [month_rank]
        })
        new_features_scaled = scaler_D.transform(new_features)
        predicted = model_D_knn.predict(new_features_scaled)[0]
        return (predicted - capacities_D) ** 2
    
    # Set reasonable bounds for the fare (adjust as needed)
    lower_bound = 10    # minimum fare
    upper_bound = 500   # maximum fare
    
    # Run the optimizer using bounded method
    res = minimize_scalar(objective, bounds=(lower_bound, upper_bound), method='bounded')
    optimal_fare = res.x
    
    # Get the corresponding predicted pax using the optimal fare
    new_features = pd.DataFrame({
        'avg_fare_D': [optimal_fare],
        'Selling Prices': [selling_prices],
        'capacities_D': [capacities_D],
        'Month_Rank': [month_rank]
    })
    new_features_scaled = scaler_D.transform(new_features)
    optimal_predicted = model_D_knn.predict(new_features_scaled)[0]
    
    # Compute the effective load factor (capped at 100%)
    effective_LF = min(optimal_predicted, capacities_D) / capacities_D
    
    print("\nFINAL OPTIMIZATION RESULTS (Domestic):")
    print(f"Optimal avg_fare_D: {optimal_fare:.2f}")
    print(f"Optimal predicted pax: {optimal_predicted:.0f}")
    print(f"Capacity: {capacities_D:.0f}")
    print(f"Effective Load Factor: {effective_LF*100:.2f}%")
    

def interactive_prediction():
    print("\n" + "="*50)
    print("INTERACTIVE NEW MONTH PREDICTION")
    print("="*50)
    try:
        avg_fare_D = float(input("Enter avg_fare_D for domestic (e.g., 27): "))
        avg_fare_I = float(input("Enter avg_fare_I for international (e.g., 72): "))
        selling_prices = float(input("Enter Selling Prices (e.g., 103.13): "))
        capacities_D = float(input("Enter capacities_D for domestic (e.g., 99837.0): "))
        capacities_I = float(input("Enter capacities_I for international (e.g., 110000.0): "))
        month_rank = float(input("Enter Month_Rank (e.g., 10): "))
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