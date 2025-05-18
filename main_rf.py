import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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
df = df.drop(['Date', 'Unnamed: 0', 'region_encoded'], axis=1, errors='ignore')

# Features and target
X = df.drop('AveragePrice', axis=1)
Y = df['AveragePrice']
# One-hot encode 'region' and 'type'
categorical_features = ['region', 'type']
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False), categorical_features)],
    remainder='passthrough'
)
X_encoded = ct.fit_transform(X)

# Features and target
X = df.drop('AveragePrice', axis=1)
Y = df['AveragePrice']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

print('R2 score:', r2_score(Y_test, y_pred)) 