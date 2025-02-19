
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# Step 1: Load and Inspect the Data
# ===============================
df = pd.read_csv("Data Files/deepthink_data.csv")
print("="*50)
print("DATA OVERVIEW")
print("="*50)
print("First few rows of the dataset:")
print(df.head())

# Calculate the average airplane capacity for domestic and international flights
df['average_airplane_capacity_D'] = df['seats_D'] / df['flights_D']
df['average_airplane_capacity_I'] = df['seats_I'] / df['flights_I']

average_airplane_capacity_D = df['average_airplane_capacity_D'].mean()
average_airplane_capacity_I = df['average_airplane_capacity_I'].mean()

print("\nOverall Average Airplane Capacity (Domestic): {:.2f}".format(average_airplane_capacity_D))
print("Overall Average Airplane Capacity (International): {:.2f}".format(average_airplane_capacity_I))

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
# Step 4: Train Random Forest Regression Models with Given Parameters
# ===============================
# Using the best parameters for Domestic model:
# {'max_depth': 7, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}
rf_params = {
    'n_estimators': 150,
    'max_depth': 7,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42
}

# Instantiate models using the given parameters (using same settings for International model here)
rf_model_D = RandomForestRegressor(**rf_params)
rf_model_I = RandomForestRegressor(**rf_params)

# Train the models
rf_model_D.fit(X_train_D, y_train_D)
rf_model_I.fit(X_train_I, y_train_I)

# Make predictions on the test sets
y_pred_D = rf_model_D.predict(X_test_D)
y_pred_I = rf_model_I.predict(X_test_I)

print("\n" + "="*50)
print("PREDICTIONS (Random Forest Regression with Fixed Parameters)")
print("="*50)
print("Domestic Predictions:")
print([float(x) for x in y_pred_D])
print("Domestic Actuals:")
print(y_test_D.tolist())

print("\nInternational Predictions:")
print([float(x) for x in y_pred_I])
print("International Actuals:")
print(y_test_I.tolist())

# ===============================
# Step 5: Model Evaluation using Cross-Validation
# ===============================
cv_folds = 8

cv_scores_D = cross_val_score(rf_model_D, df[features_D], df[target_pax_D],
                              cv=cv_folds, scoring='r2')
cv_scores_I = cross_val_score(rf_model_I, df[features_I], df[target_pax_I],
                              cv=cv_folds, scoring='r2')

print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)
print("Domestic R² Scores per Fold:", cv_scores_D)
print(f"Domestic Average R² Score: {cv_scores_D.mean():.2f}")

print("\nInternational R² Scores per Fold:", cv_scores_I)
print(f"International Average R² Score: {cv_scores_I.mean():.2f}")

mae_scores_D = cross_val_score(rf_model_D, df[features_D], df[target_pax_D],
                               cv=cv_folds, scoring='neg_mean_absolute_error')
mae_scores_I = cross_val_score(rf_model_I, df[features_I], df[target_pax_I],
                               cv=cv_folds, scoring='neg_mean_absolute_error')

print("\nDomestic MAE per Fold (in absolute value):", [float(-score) for score in mae_scores_D])
print(f"Domestic Average MAE: {-mae_scores_D.mean():.2f}")

print("\nInternational MAE per Fold (in absolute value):", [float(-score) for score in mae_scores_I])
print(f"International Average MAE: {-mae_scores_I.mean():.2f}")

# ===============================
# Step 6: Plotting Curves to Visualize Performance
# ===============================

# 6.1 Learning Curve for the Domestic Model (using R² as the metric)
train_sizes, train_scores, val_scores = learning_curve(
    rf_model_D, df[features_D], df[target_pax_D],
    cv=cv_folds, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training R²")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation R²")
plt.title("Learning Curve for Domestic Random Forest Regression")
plt.xlabel("Training Examples")
plt.ylabel("R² Score")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# 6.2 Residual Plot for the Domestic Model
y_train_pred = rf_model_D.predict(X_train_D)
residuals = y_train_D - y_train_pred

plt.figure(figsize=(10,6))
plt.scatter(y_train_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot for Domestic Random Forest Regression")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.show()

# ===============================
# Step 7: Optional - Interactive New Month Prediction
# ===============================
def interactive_prediction_rf():
    print("\n" + "="*50)
    print("INTERACTIVE NEW MONTH PREDICTION (Random Forest with Fixed Parameters)")
    print("="*50)
    try:
        avg_fare_D = float(input("Enter the Average Fare for Domestic (e.g. 27): "))
        avg_fare_I = float(input("Enter the Average Fare for International (e.g. 72): "))
        selling_prices = float(input("Enter Selling Prices (e.g. 103.13): "))
        capacities_D = float(input("Enter Capacities for Domestic (e.g. 99837.0): "))
        capacities_I = float(input("Enter Capacities for International (e.g. 110000.0): "))
        month_rank = float(input("Enter the Month's Rank (e.g. 10): "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return

    new_features_D = pd.DataFrame([[avg_fare_D, selling_prices, capacities_D, month_rank]],
                                  columns=features_D)
    new_features_I = pd.DataFrame([[avg_fare_I, selling_prices, capacities_I, month_rank]],
                                  columns=features_I)

    pred_pax_D = rf_model_D.predict(new_features_D)[0]
    pred_pax_I = rf_model_I.predict(new_features_I)[0]

    print("\n--- Prediction Results ---")
    print(f"Predicted Domestic Passengers (pax_D): {pred_pax_D:.0f}")
    print(f"Predicted International Passengers (pax_I): {pred_pax_I:.0f}")

interactive_prediction_rf()
