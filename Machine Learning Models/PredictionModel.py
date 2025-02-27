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

# Compute weighted competitor price using historical lags
def compute_weighted_competitor_price(df, flight_type: FlightType, target_year, target_month, weights=(0.6, 0.3, 0.1)):
    competitor_col = f"competitors_price_{flight_type.value}"
    competitor_values = []
    for lag in range(1, 4):
        value = df.loc[(df['year'] == target_year - lag) & (df['month'] == target_month), competitor_col]
        if not value.empty:
            competitor_values.append(value.iloc[0])
        else:
            competitor_values.append(np.nan)
    competitor_values = np.array(competitor_values, dtype=float)
    if np.isnan(competitor_values).any():
        competitor_values = np.where(np.isnan(competitor_values), np.nanmean(competitor_values), competitor_values)
    weighted_price = (competitor_values[0] * weights[0] +
                      competitor_values[1] * weights[1] +
                      competitor_values[2] * weights[2])
    return weighted_price

def get_user_data(flight_type: FlightType):
    try:
        year = int(input("Enter year: "))
        month = int(input("Enter month (1-12): "))
        avg_fare = float(input(f"Enter avg_fare_{flight_type.value} (ticket price): "))
        # FIX 1: Ask for weighted competitor selling price instead of raw competitor price
        competitor_price = float(input("Enter weighted competitor selling price: "))
        seats = float(input(f"Enter seats_{flight_type.value} (available seats): "))
        load_factor = float(input(f"Enter LF_{flight_type.value} (load factor, e.g., 0-100): "))
        return year, month, avg_fare, competitor_price, seats, load_factor
    except ValueError:
        print("Please enter valid numeric inputs.")
        return None

def predict_passengers(flight_type: FlightType, filter_month=None):
    # Load the data
    df = pd.read_csv('Data Files/deepthink_data_v2.csv')
    
    competitor_col = f"competitors_price_{flight_type.value}"
    if competitor_col not in df.columns:
        raise KeyError(f"Column not found: {competitor_col}")
    
    df['year'] = df['year'].astype(int)
    df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

    # FIX 3: Sort the data by year and month before computing lag features
    df = df.sort_values(by=['year', 'month'])
    
    if filter_month is not None:
        df = df[df['month'] == filter_month]

    # Create lagged competitor selling prices columns (grouping by month)
    df['selling_prices_lag1'] = df.groupby('month')[competitor_col].shift(1)
    df['selling_prices_lag2'] = df.groupby('month')[competitor_col].shift(2)
    df['selling_prices_lag3'] = df.groupby('month')[competitor_col].shift(3)
    
    # Calculate weighted competitor selling price using weights 0.6, 0.3, 0.1
    df['weighted_selling_prices'] = (df['selling_prices_lag1'] * 0.6 +
                                     df['selling_prices_lag2'] * 0.3 +
                                     df['selling_prices_lag3'] * 0.1)
    # Avoid chained assignment warning:
    df['weighted_selling_prices'] = df['weighted_selling_prices'].fillna(df[competitor_col])
    
    features = ['year', 'month', f'avg_fare_{flight_type.value}', 'weighted_selling_prices',
                f'seats_{flight_type.value}', f'LF_{flight_type.value}']
    X = df[features].copy()
    y = df[f'pax_{flight_type.value}']
    
    X[f'avg_fare_{flight_type.value}'] = np.log1p(X[f'avg_fare_{flight_type.value}'])
    y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=64)

    def grid_search_and_fit(model, params, X_train, y_train):
        grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Best parameters for {model.__class__.__name__}: {grid.best_params_}")
        return grid.best_estimator_

    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'gamma': [0, 0.1, 0.3]
    }
    xgb_model = grid_search_and_fit(XGBRegressor(random_state=64), xgb_params, X_train, y_train)

    ridge_params = {
        'alpha': [0.001, 0.01, 0.1, 1, 10]
    }
    ridge_model = grid_search_and_fit(Ridge(random_state=64), ridge_params, X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn_params = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
    knn_model = grid_search_and_fit(KNeighborsRegressor(), knn_params, X_train_scaled, y_train)

    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    rf_model = grid_search_and_fit(RandomForestRegressor(random_state=64), rf_params, X_train, y_train)

    models = {
        'XGBoost': xgb_model,
        'Ridge': ridge_model,
        'kNN': knn_model,
        'RandomForest': rf_model
    }

    r2_scores_train = {}
    r2_scores_test = {}
    for name, model in models.items():
        if name == 'kNN':
            r2_scores_train[name] = r2_score(y_train, model.predict(X_train_scaled))
            r2_scores_test[name] = r2_score(y_test, model.predict(X_test_scaled))
        else:
            r2_scores_train[name] = r2_score(y_train, model.predict(X_train))
            r2_scores_test[name] = r2_score(y_test, model.predict(X_test))

    ensemble_weights = {k: v / sum(r2_scores_test.values()) for k, v in r2_scores_test.items()}

    final_pred_log = predict_with_models(models, X_test, ensemble_weights, scaler)
    final_pred = np.expm1(final_pred_log)
    y_test_orig = np.expm1(y_test)

    final_pred_train_log = predict_with_models(models, X_train, ensemble_weights, scaler)
    final_pred_train = np.expm1(final_pred_train_log)
    y_train_orig = np.expm1(y_train)

    final_r2_train = r2_score(y_train_orig, final_pred_train)
    final_r2_test = r2_score(y_test_orig, final_pred)

    for i in range(len(final_pred)):
        print(f"Predicted {flight_type.name} Passengers: {final_pred[i]:.0f}, "
              f"Actual {flight_type.name} Passengers: {y_test_orig.iloc[i]:.0f}, "
              f"Difference: {final_pred[i] - y_test_orig.iloc[i]:.0f}")

    print(f"R² Scores for Individual Models ({flight_type.name}) - Log Scale:")
    for name in models.keys():
        print(f"{name}: Train R² {r2_scores_train[name]:.4f}, Test R² {r2_scores_test[name]:.4f}")

    print(f"\nWeights for {flight_type.name} Models:")
    for name, weight in ensemble_weights.items():
        print(f"{name}: {weight:.4f}")

    print(f"\nFinal R² Score for {flight_type.name} Ensemble - Original Scale - Train: {final_r2_train:.4f}, Test: {final_r2_test:.4f}")

    return models, scaler, ensemble_weights, features

def predict_new_data(flight_type: FlightType, models, scaler, weights, features):
    user_data = get_user_data(flight_type)
    if user_data:
        year, month, avg_fare, seats, load_factor = user_data
        # Load dataset and compute weighted competitor price for the target year/month
        df = pd.read_csv('Data Files/deepthink_data_v2.csv')
        df['year'] = df['year'].astype(int)
        df['month'] = pd.to_datetime(df['month'], format='%B').dt.month
        weighted_competitor_price = compute_weighted_competitor_price(df, flight_type, year, month)
        # Build input data. The feature order is: year, month, avg_fare_x, weighted_selling_prices, seats_x, LF_x
        col_avg_fare = f'avg_fare_{flight_type.value}'
        col_seats = f'seats_{flight_type.value}'
        col_lf = f'LF_{flight_type.value}'
        input_data = pd.DataFrame([(year, month, avg_fare, weighted_competitor_price, seats, load_factor)],
                                  columns=features)
        input_data[col_avg_fare] = np.log1p(input_data[col_avg_fare])
        prediction_log = predict_with_models(models, input_data, weights, scaler)
        prediction = np.expm1(prediction_log)
        return prediction[0]

def simulate_domestic_predictions_by_month():
    print("Starting simulation by month...")

    target_year = int(input("Enter target year for simulation: "))
    target_month = int(input("Enter target month (1-12) for simulation: "))

    models, scaler, weights, features = predict_passengers(FlightType.DOMESTIC, filter_month=target_month)

    df = pd.read_csv('Data Files/deepthink_data_v2.csv')
    df['year'] = df['year'].astype(int)
    df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

    competitor_values = []
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
    # FIX 2: Use the same weights as training: 0.6, 0.3, 0.1
    weighted_competitor_price = competitor_values[0] * 0.6 + competitor_values[1] * 0.3 + competitor_values[2] * 0.1
    print("Using weighted competitor price:", weighted_competitor_price)

    mean_seats = df['seats_D'].mean()
    mean_lf = df['LF_D'].mean()
    print(f"Using mean seats_D: {mean_seats:.0f}, mean LF_D: {mean_lf:.1f}")

    avg_fare_values = np.linspace(50, 200, 10)

    for avg_fare in avg_fare_values:
        input_dict = {
            'year': target_year,
            'month': target_month,
            'avg_fare_D': avg_fare,  # log transform will be applied below
            # FIX 1: Supply value for "weighted_selling_prices"
            'weighted_selling_prices': weighted_competitor_price,
            'seats_D': mean_seats,
            'LF_D': mean_lf
        }
        input_df = pd.DataFrame([input_dict], columns=features)
        input_df[f'avg_fare_D'] = np.log1p(input_df[f'avg_fare_D'])
        prediction_log = predict_with_models(models, input_df, weights, scaler)
        prediction = np.expm1(prediction_log)
        print(f"For avg_fare_D = {avg_fare:.2f}, predicted domestic passengers = {prediction[0]:.0f}")

def main(year, month, avg_fare_D, avg_fare_I, competitors_price_D, competitors_price_I, flight_type_str):
    # Train models for both domestic and international flights
    domestic_models, domestic_scaler, domestic_weights, domestic_features = predict_passengers(FlightType.DOMESTIC)
    international_models, international_scaler, international_weights, international_features = predict_passengers(FlightType.INTERNATIONAL)

    flight_type = FlightType[flight_type_str.upper()]

    if flight_type == FlightType.DOMESTIC:
        df = pd.read_csv('Data Files/deepthink_data_v2.csv')
        mean_seats_D = df['seats_D'].mean()
        mean_lf_D = df['LF_D'].mean()
        # FIX 1: Use the provided competitors_price_D as the weighted competitor price input
        user_data = (year, month, avg_fare_D, competitors_price_D, mean_seats_D, mean_lf_D)
        input_data = pd.DataFrame([user_data], columns=domestic_features)
        input_data['avg_fare_D'] = np.log1p(input_data['avg_fare_D'])
        prediction_log = predict_with_models(domestic_models, input_data, domestic_weights, domestic_scaler)
        return np.expm1(prediction_log)[0]
    elif flight_type == FlightType.INTERNATIONAL:
        weighted_competitor_price = compute_weighted_competitor_price(df, FlightType.INTERNATIONAL, year, month)
        mean_seats_I = df['seats_I'].mean()
        mean_lf_I = df['LF_I'].mean()
        user_data = (year, month, avg_fare_I, competitors_price_I, mean_seats_I, mean_lf_I)
        input_data = pd.DataFrame([user_data], columns=international_features)
        input_data['avg_fare_I'] = np.log1p(input_data['avg_fare_I'])
        prediction_log = predict_with_models(international_models, input_data, international_weights, international_scaler)
        return np.expm1(prediction_log)[0]

if __name__ == "__main__":
    year = 2024
    month = 8
    avg_fare_D = 40.0       # Domestic average fare
    avg_fare_I = 150.0      # International average fare (example value)
    competitors_price_D = 50.0  # For predictions, assume this is the weighted competitor price
    competitors_price_I = 260.0  # International competitor price (example value)
    flight_type_str = 'DOMESTIC'  # 'DOMESTIC' or 'INTERNATIONAL'
    print(main(year, month, avg_fare_D, avg_fare_I, competitors_price_D, competitors_price_I, flight_type_str))
