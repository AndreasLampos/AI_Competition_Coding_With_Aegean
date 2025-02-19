import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from enum import Enum

class FlightType(Enum):
    DOMESTIC = 'D'
    INTERNATIONAL = 'I'

def predict_passengers(flight_type: FlightType):
    # Load the data
    df = pd.read_csv('Data Files/deepthink_data.csv')

    # Feature engineering
    df['year'] = df['year'].astype(int)
    df['month'] = pd.to_datetime(df['month'], format='%B').dt.month  # Convert month to numeric

    # Define features and target
    features = ['year', 'month_rank', f'avg_fare_{flight_type.value}', 'selling_prices', f'capacities_{flight_type.value}']
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

    # Linear Regression with Ridge
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

    # Train models and compute R² scores for both train and test
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

    # Calculate weights based on test R² scores
    weights = {k: v / sum(r2_scores_test.values()) for k, v in r2_scores_test.items()}

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

    # Final predictions for test data
    final_pred = predict_with_models(models, X_test, weights, scaler)

    # Calculate final R² scores for ensemble for both train and test
    final_pred_train = predict_with_models(models, X_train, weights, scaler)
    final_r2_train = r2_score(y_train, final_pred_train)
    final_r2_test = r2_score(y_test, final_pred)

    for i in range(len(final_pred)):
        print(f"Predicted {flight_type.name} Passengers: {final_pred[i]:.0f}, Actual {flight_type.name} Passengers: {y_test.iloc[i]:.0f}, Difference: {final_pred[i] - y_test.iloc[i]:.0f}")

    print(f"R² Scores for Individual Models ({flight_type.name}):")
    for name in models.keys():
        print(f"{name}: Train R² {r2_scores_train[name]:.4f}, Test R² {r2_scores_test[name]:.4f}")

    print(f"\nWeights for {flight_type.name} Models:")
    for name, weight in weights.items():
        print(f"{name}: {weight:.4f}")

    print(f"\nFinal R² Score for {flight_type.name} Ensemble - Train: {final_r2_train:.4f}, Test: {final_r2_test:.4f}")

    return final_pred

# Example usage
domestic_predictions = predict_passengers(FlightType.DOMESTIC)
international_predictions = predict_passengers(FlightType.INTERNATIONAL)