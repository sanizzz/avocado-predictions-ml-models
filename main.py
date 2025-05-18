import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load data
csv_path = "avocado1.csv"
df = pd.read_csv(csv_path)

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['dayofweek'] = df['Date'].dt.dayofweek

# Drop unnecessary columns
columns_to_drop = [
    '4046', 'Unnamed: 0', '4225', '4770',          # specific product codes
    'Small Bags', 'Large Bags', 'XLarge Bags',     # bag sizes
    'Date'                                          # not used directly
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Features and target
X = df.drop('AveragePrice', axis=1)
Y = df['AveragePrice']

# One-hot encode 'region' and 'type' using ColumnTransformer
categorical_features = ['region', 'type']
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False), categorical_features)],
    remainder='passthrough'
)
X_encoded = ct.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVR requires dependent var to be 1D
regressor = SVR(kernel='rbf')
regressor.fit(X_train, Y_train)  # No need for ravel() since Y_train is already 1D use of ravel is svm requires dependant var to be 1d so we flatten it using .ravel()

# Make predictions
y_pred = regressor.predict(X_test)

# Calculate and print R2 score
r2 = r2_score(Y_test, y_pred)
print(f'R2 score: {r2:.4f}')

# Print additional metrics to better understand model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}') 