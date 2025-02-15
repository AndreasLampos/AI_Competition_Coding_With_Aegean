import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('merged_set.csv')

# Print the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Define features and target variables
features_D = ['avg_fare_D', 'Selling Prices ', 'Capacities ', 'Month_Rank']
features_I = ['avg_fare_I', 'Selling Prices ', 'Capacities ', 'Month_Rank']
target_pax_D = 'pax_D'
target_pax_I = 'pax_I'

# Print the summary statistics of the features and target variables
print("\nSummary statistics of the features and target variables:")
print(df[features_D + features_I + [target_pax_D, target_pax_I]].describe())

# Split the data into training and testing sets for pax_D
X_train_D, X_test_D, y_train_D, y_test_D = train_test_split(df[features_D], df[target_pax_D], test_size=0.2, random_state=42)

# Split the data into training and testing sets for pax_I
X_train_I, X_test_I, y_train_I, y_test_I = train_test_split(df[features_I], df[target_pax_I], test_size=0.2, random_state=42)

# Print the lengths of the training and testing sets
print(f"\nNumber of training samples for pax_D: {len(X_train_D)}")
print(f"Number of testing samples for pax_D: {len(X_test_D)}")
print(f"Number of training samples for pax_I: {len(X_train_I)}")
print(f"Number of testing samples for pax_I: {len(X_test_I)}")

# Standardize the features
scaler = StandardScaler()
X_train_D = scaler.fit_transform(X_train_D)
X_test_D = scaler.transform(X_test_D)
X_train_I = scaler.fit_transform(X_train_I)
X_test_I = scaler.transform(X_test_I)

# Print the first few rows of the standardized features
print("\nFirst few rows of the standardized features for pax_D:")
print(X_train_D[:5])
print("\nFirst few rows of the standardized features for pax_I:")
print(X_train_I[:5])

# Function to calculate and print MAE and Bias
def evaluate_model(y_true, y_pred, model_name, target_name):
    mae = mean_absolute_error(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    print(f'\n{model_name} {target_name} - Mean Absolute Error: {mae:.2f}, Bias: {bias:.2f}, MAE + |Bias|: {mae + abs(bias):.2f}')

# Initialize and train the Linear Regression model for pax_D
model_D_lr = LinearRegression()
model_D_lr.fit(X_train_D, y_train_D)

# Make predictions for pax_D using Linear Regression
y_pred_D_lr = model_D_lr.predict(X_test_D)

# Print all predictions and actual values for pax_D
print("\nPredictions for pax_D using Linear Regression:")
print(list(y_pred_D_lr))
print("\nActual values for pax_D:")
print(list(y_test_D))

# Evaluate the Linear Regression model for pax_D
evaluate_model(y_test_D, y_pred_D_lr, 'Linear Regression', 'pax_D')

# Initialize and train the Linear Regression model for pax_I
model_I_lr = LinearRegression()
model_I_lr.fit(X_train_I, y_train_I)

# Make predictions for pax_I using Linear Regression
y_pred_I_lr = model_I_lr.predict(X_test_I)

# Print all predictions and actual values for pax_I
print("\nPredictions for pax_I using Linear Regression:")
print(list(y_pred_I_lr))
print("\nActual values for pax_I:")
print(list(y_test_I))

# Evaluate the Linear Regression model for pax_I
evaluate_model(y_test_I, y_pred_I_lr, 'Linear Regression', 'pax_I')