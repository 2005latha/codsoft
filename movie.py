import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Create a sample dataset
data = pd.DataFrame({
    'feature1': [5,3,5,2,6,],
    'feature2': [1,2,3,4,5],
    'genre': ['Action', 'Comedy', 'Action', 'Comedy', 'Action'],
    'rating': [3.2, 4.5, 3.8, 2.0, 3.5]
})

# Step 2: Data preparation
data = data.dropna()  # Handle missing values
data = pd.get_dummies(data, columns=['genre'], drop_first=True)  # One-hot encoding

# Step 3: Define features and target variable
X = data.drop(['rating'], axis=1)  # Features
y = data['rating']  # Target variable

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# Step 7: Making predictions
# Ensure new_data has the same features as X
new_data = pd.DataFrame({
    'feature1': [3],
    'feature2': [5],
    'genre_Comedy': [0],
    'genre_Action': [1]
})

# Ensure consistent feature columns
new_data = new_data.reindex(columns=X.columns, fill_value=0)

predicted_rating = model.predict(new_data)
print(f'Predicted Rating: {predicted_rating[0]:.2f}')
