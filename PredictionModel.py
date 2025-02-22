import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from enum import Enum
import price


class FlightType(Enum):
    DOMESTIC = 'D'
    INTERNATIONAL = 'I'

# Predict using all models
def predict_with_models(models, X_data, weights, scaler=None):
    predictions = []
    for name, model in models.items():
        if name == 'kNN':
            pred = model.predict(scaler.transform(X_data))
        else:
            pred = model.predict(X_data)
        predictions.append(pred * weights[name])
    return np.sum(predictions, axis=0)

def get_user_data(flight_type: FlightType):
    try:
        year = int(input("Enter year: "))
        month = int(input("Enter month (1-12): "))
        avg_fare = float(input(f"Enter avg_fare_{flight_type.value} (ticket price): "))
        competitor_price = float(input("Enter competitor selling price: "))
        # Order must match features: ['year', 'month', f'avg_fare_{flight_type.value}', 'weighted_selling_prices']
        return year, month, avg_fare, competitor_price
    except ValueError:
        print("Please enter valid numeric inputs.")
        return None

def predict_passengers(flight_type: FlightType, filter_month=None):
    # Load the data
    df = pd.read_csv('Data Files/deepthink_data_v2.csv')
    
    # Use the correct competitor price column based on flight type
    competitor_col = f"competitors_price_{flight_type.value}"
    if competitor_col not in df.columns:
        raise KeyError(f"Column not found: {competitor_col}")
    
    # Convert year and month (assuming month is stored as full month name)
    df['year'] = df['year'].astype(int)
    df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

    # Filter by month, if provided
    if filter_month is not None:
        df = df[df['month'] == filter_month]

    # Create lagged competitor selling prices columns (using the correct column)
    df['selling_prices_lag1'] = df.groupby('month')[competitor_col].shift(1)
    df['selling_prices_lag2'] = df.groupby('month')[competitor_col].shift(2)
    df['selling_prices_lag3'] = df.groupby('month')[competitor_col].shift(3)
    
    # Calculate weighted competitor selling price using weights 0.6, 0.3, 0.1
    df['weighted_selling_prices'] = (df['selling_prices_lag1'] * 0.6 +
                                     df['selling_prices_lag2'] * 0.3 +
                                     df['selling_prices_lag3'] * 0.1)
    # For rows with missing lagged values, fill with the current competitor price column value
    df['weighted_selling_prices'].fillna(df[competitor_col], inplace=True)
    
    # Define features and target (using the appropriate avg_fare and pax columns)
    features = ['year', 'month', f'avg_fare_{flight_type.value}', 'weighted_selling_prices']
    X = df[features]
    y = df[f'pax_{flight_type.value}']
    

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

    # Function for grid search and fitting
    def grid_search_and_fit(model, params, X_train, y_train):
        grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Best parameters for {model.__class__.__name__}: {grid.best_params_}")
        return grid.best_estimator_

    # XGBoost
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'gamma': [0, 0.1, 0.3]
    }
    xgb_model = grid_search_and_fit(XGBRegressor(random_state=64), xgb_params, X_train, y_train)

    # Ridge Regression
    ridge_params = {
        'alpha': [0.001, 0.01, 0.1, 1, 10]
    }
    ridge_model = grid_search_and_fit(Ridge(random_state=64), ridge_params, X_train, y_train)

    # kNN with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn_params = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
    knn_model = grid_search_and_fit(KNeighborsRegressor(), knn_params, X_train_scaled, y_train)

    # Random Forest
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    rf_model = grid_search_and_fit(RandomForestRegressor(random_state=64), rf_params, X_train, y_train)

    # Prepare models dictionary
    models = {
        'XGBoost': xgb_model,
        'Ridge': ridge_model,
        'kNN': knn_model,
        'RandomForest': rf_model
    }

    # Compute R² scores for each model for train and test sets
    r2_scores_train = {}
    r2_scores_test = {}
    for name, model in models.items():
        if name == 'kNN':
            r2_scores_train[name] = r2_score(y_train, model.predict(X_train_scaled))
            r2_scores_test[name] = r2_score(y_test, model.predict(X_test_scaled))
        else:
            r2_scores_train[name] = r2_score(y_train, model.predict(X_train))
            r2_scores_test[name] = r2_score(y_test, model.predict(X_test))

    # Calculate ensemble weights based on test R² scores
    ensemble_weights = {k: v / sum(r2_scores_test.values()) for k, v in r2_scores_test.items()}

    # Ensemble predictions on test set
    final_pred = predict_with_models(models, X_test, ensemble_weights, scaler)

    # Ensemble predictions for training set
    final_pred_train = predict_with_models(models, X_train, ensemble_weights, scaler)
    final_r2_train = r2_score(y_train, final_pred_train)
    final_r2_test = r2_score(y_test, final_pred)

    # Print predictions and scores
    for i in range(len(final_pred)):
        print(f"Predicted {flight_type.name} Passengers: {final_pred[i]:.0f}, Actual {flight_type.name} Passengers: {y_test.iloc[i]:.0f}, Difference: {final_pred[i] - y_test.iloc[i]:.0f}")

    print(f"R² Scores for Individual Models ({flight_type.name}):")
    for name in models.keys():
        print(f"{name}: Train R² {r2_scores_train[name]:.4f}, Test R² {r2_scores_test[name]:.4f}")

    print(f"\nWeights for {flight_type.name} Models:")
    for name, weight in ensemble_weights.items():
        print(f"{name}: {weight:.4f}")

    print(f"\nFinal R² Score for {flight_type.name} Ensemble - Train: {final_r2_train:.4f}, Test: {final_r2_test:.4f}")

    return models, scaler, ensemble_weights, features

def predict_new_data(flight_type: FlightType, models, scaler, weights, features):
    user_data = get_user_data(flight_type)
    if user_data:
        input_data = pd.DataFrame([user_data], columns=features)
        prediction = predict_with_models(models, input_data, weights, scaler)

        return prediction[0]
    
def simulate_domestic_predictions_by_month():
    print("Starting simulation by month...")

    # Prompt user for target simulation year and month
    target_year = int(input("Enter target year for simulation: "))
    target_month = int(input("Enter target month (1-12) for simulation: "))

    # Train a model using only data from the target month
    models, scaler, weights, features = predict_passengers(FlightType.DOMESTIC, filter_month=target_month)

    # Load dataset (again) to extract competitor prices for the target month from previous 3 years
    df = pd.read_csv('Data Files/deepthink_data_v2.csv')
    df['year'] = df['year'].astype(int)
    df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

    competitor_values = []
    # Use the appropriate competitor price column for domestic, i.e. "competitors_price_D"
    competitor_col = "competitors_price_D"
    for lag in range(1, 4):
        value = df.loc[(df['year'] == target_year - lag) & (df['month'] == target_month), competitor_col]
        if not value.empty:
            competitor_values.append(value.iloc[0])
        else:
            competitor_values.append(np.nan)
    competitor_values = np.array(competitor_values, dtype=float)
    if np.isnan(competitor_values).any():
        competitor_values = np.where(np.isnan(competitor_values), np.nanmean(competitor_values), competitor_values)

    # Compute weighted competitor price using weights: 0.5, 0.3, 0.2
    weighted_competitor_price = competitor_values[0] * 0.5 + competitor_values[1] * 0.3 + competitor_values[2] * 0.2
    print("Using weighted competitor price:", weighted_competitor_price)

    # Define a range of avg_fare values to test (ensure these are within your training range)
    avg_fare_values = np.linspace(50, 200, 10)

    for avg_fare in avg_fare_values:
        input_dict = {
            'year': target_year,
            'month': target_month,
            f'avg_fare_D': avg_fare,
            'weighted_selling_prices': weighted_competitor_price
        }   
        input_df = pd.DataFrame([input_dict], columns=features)
        prediction = predict_with_models(models, input_df, weights, scaler)
        print(f"For avg_fare_D = {avg_fare:.2f}, predicted domestic passengers = {prediction[0]:.0f}")

def main(year, month, avg_fare, selling_prices, capacities, flight_type_str):
    # Train models for both domestic and international flights
    domestic_models, domestic_scaler, domestic_weights, domestic_features = predict_passengers(FlightType.DOMESTIC)
    international_models, international_scaler, international_weights, international_features = predict_passengers(FlightType.INTERNATIONAL)

    flight_type = FlightType[flight_type_str.upper()]

    # Predict for new data based on chosen flight type
    if flight_type == FlightType.DOMESTIC:
        simulate_domestic_predictions_by_month()
    elif flight_type == FlightType.INTERNATIONAL:
        return predict_new_data(year, month, avg_fare, selling_prices, capacities, FlightType.INTERNATIONAL, international_models, international_scaler, international_weights, international_features)

if __name__ == "__main__":
    # Example usage
    year = 2025
    month = 8
    avg_fare = 300.0
    selling_prices = 350.0
    capacities = 150.0
    flight_type_str = 'DOMESTIC'
    print(main(year, month, avg_fare, selling_prices, capacities, flight_type_str))