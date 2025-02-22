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

# Define flight types
class FlightType(Enum):
    DOMESTIC = 'D'
    INTERNATIONAL = 'I'

# Predict using ensemble of models
def predict_with_models(models, X_data, weights, scaler=None):
    predictions = []
    for name, model in models.items():
        if name == 'kNN':
            pred = model.predict(scaler.transform(X_data))
        else:
            pred = model.predict(X_data)
        predictions.append(pred * weights[name])
    return np.sum(predictions, axis=0)

# Adjust passenger forecast based on price elasticity
def adjust_passengers(pax_forecast, avg_fare, competitor_price, month_rank, sensitivity=2):
    diff = avg_fare - competitor_price
    relative_diff = diff / competitor_price
    dampening = 1 / month_rank

    if diff > 0:
        adjustment_factor = 1 - sensitivity * relative_diff * dampening
    elif diff < 0:
        adjustment_factor = 1 + sensitivity * abs(relative_diff) * dampening
    else:
        adjustment_factor = 1.0

    adjusted_forecast = pax_forecast * adjustment_factor
    return max(adjusted_forecast, 0)

# Modified grid_search_and_fit to use best_params if provided
def grid_search_and_fit(model, X_train, y_train, best_params=None, params=None):
    """
    Fits a model using either predefined best parameters or performs grid search.
    
    Args:
        model: The machine learning model instance.
        X_train: Training feature data.
        y_train: Training target data.
        best_params: Optional dictionary of best parameters to set directly.
        params: Optional dictionary of parameters for grid search.
    
    Returns:
        The fitted model.
    """
    if best_params:
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        return model
    elif params:
        grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Best parameters for {model.__class__.__name__}: {grid.best_params_}")
        return grid.best_estimator_
    else:
        raise ValueError("Either best_params or params must be provided.")

# Train models and prepare for prediction
def predict_passengers(flight_type: FlightType):
    df = pd.read_csv('Data Files/deepthink_data_v2.csv')
    
    competitor_col = f"competitors_price_{flight_type.value}"
    if competitor_col not in df.columns:
        raise KeyError(f"Column not found: {competitor_col}")
    
    df['year'] = df['year'].astype(int)
    df['month'] = pd.to_datetime(df['month'], format='%B').dt.month

    # Create lagged competitor selling prices
    df['selling_prices_lag1'] = df.groupby('month')[competitor_col].shift(1)
    df['selling_prices_lag2'] = df.groupby('month')[competitor_col].shift(2)
    df['selling_prices_lag3'] = df.groupby('month')[competitor_col].shift(3)
    
    # Calculate weighted selling prices
    df['weighted_selling_prices'] = (df['selling_prices_lag1'] * 0.6 +
                                     df['selling_prices_lag2'] * 0.3 +
                                     df['selling_prices_lag3'] * 0.1)
    df['weighted_selling_prices'].fillna(df[competitor_col], inplace=True)
    
    # Define features and target
    features = ['year', 'month', f'avg_fare_{flight_type.value}', 'weighted_selling_prices',
                f'seats_{flight_type.value}', f'LF_{flight_type.value}']
    X = df[features].copy()
    y = df[f'pax_{flight_type.value}']
    
    # Log-transform skewed variables
    X[f'avg_fare_{flight_type.value}'] = np.log1p(X[f'avg_fare_{flight_type.value}'])
    y = np.log1p(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=64)

    # Define best parameters as provided
    best_params_xgb = {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300}
    best_params_ridge = {'alpha': 1}
    best_params_knn = {'n_neighbors': 3, 'weights': 'distance'}
    best_params_rf = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

    # Initialize and fit models with best parameters
    xgb_model = grid_search_and_fit(XGBRegressor(random_state=64), X_train, y_train, best_params=best_params_xgb)
    ridge_model = grid_search_and_fit(Ridge(random_state=64), X_train, y_train, best_params=best_params_ridge)
    
    # For kNN, use scaled data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn_model = grid_search_and_fit(KNeighborsRegressor(), X_train_scaled, y_train, best_params=best_params_knn)
    
    rf_model = grid_search_and_fit(RandomForestRegressor(random_state=64), X_train, y_train, best_params=best_params_rf)

    # Ensemble models
    models = {
        'XGBoost': xgb_model,
        'Ridge': ridge_model,
        'kNN': knn_model,
        'RandomForest': rf_model
    }

    # Calculate R² scores for weights
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

    return models, scaler, ensemble_weights, features

# Main prediction function
def main(year, month, avg_fare_D, avg_fare_I, competitors_price_D, competitors_price_I, flight_type_str):
    month_ranks = {
        1: 10, 2: 12, 3: 11, 4: 9, 5: 6, 6: 4, 7: 3, 8: 1, 9: 2, 10: 7, 11: 5, 12: 8
    }
    
    domestic_models, domestic_scaler, domestic_weights, domestic_features = predict_passengers(FlightType.DOMESTIC)
    international_models, international_scaler, international_weights, international_features = predict_passengers(FlightType.INTERNATIONAL)
    flight_type = FlightType[flight_type_str.upper()]
    
    if flight_type == FlightType.DOMESTIC:
        df = pd.read_csv('Data Files/deepthink_data_v2.csv')
        mean_seats_D = df['seats_D'].mean()
        mean_lf_D = df['LF_D'].mean()
        user_data = (year, month, avg_fare_D, competitors_price_D, mean_seats_D, mean_lf_D)
        input_data = pd.DataFrame([user_data], columns=domestic_features)
        input_data['avg_fare_D'] = np.log1p(input_data['avg_fare_D'])
        prediction_log = predict_with_models(domestic_models, input_data, domestic_weights, domestic_scaler)
        initial_pax = np.expm1(prediction_log)[0]
        
        month_rank = month_ranks[month]
        adjusted_pax = adjust_passengers(initial_pax, avg_fare_D, competitors_price_D, month_rank)
        
        print(f"Initial predicted {flight_type.name} passengers: {initial_pax:.0f}")
        print(f"Adjusted predicted {flight_type.name} passengers: {adjusted_pax:.0f}")
        return adjusted_pax
    
    elif flight_type == FlightType.INTERNATIONAL:
        df = pd.read_csv('Data Files/deepthink_data_v2.csv')
        mean_seats_I = df['seats_I'].mean()
        mean_lf_I = df['LF_I'].mean()
        user_data = (year, month, avg_fare_I, competitors_price_I, mean_seats_I, mean_lf_I)
        input_data = pd.DataFrame([user_data], columns=international_features)
        input_data['avg_fare_I'] = np.log1p(input_data['avg_fare_I'])
        prediction_log = predict_with_models(international_models, input_data, international_weights, international_scaler)
        initial_pax = np.expm1(prediction_log)[0]
        
        month_rank = month_ranks[month]
        adjusted_pax = adjust_passengers(initial_pax, avg_fare_I, competitors_price_I, month_rank)
        
        print(f"Initial predicted {flight_type.name} passengers: {initial_pax:.0f}")
        print(f"Adjusted predicted {flight_type.name} passengers: {adjusted_pax:.0f}")
        return adjusted_pax

# Example usage
if __name__ == "__main__":
    year = 2024
    month = 8
    avg_fare_D = 70.0
    avg_fare_I = 150.0
    competitors_price_D = 50.0
    competitors_price_I = 260.0
    flight_type_str = 'DOMESTIC'
    result = main(year, month, avg_fare_D, avg_fare_I, competitors_price_D, competitors_price_I, flight_type_str)
    print(f"Final result: {result:.0f}")