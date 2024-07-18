import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Create a sample dataset
data = pd.DataFrame({
    'month': [5, 2, 6, 6, 7, 8, 9, 9, 9],
    'feature1': [50, 100, 150, 200, 250, 300, 350, 400,450],
    'feature2': [5, 10, 15, 24, 25, 30, 34, 35, 40],
    'sales': [2000, 2300, 2500, 4800, 2000, 3300, 8500, 3700, 4000]
})

# Step 2: Data preparation
data = data.dropna()
X = data.drop(['sales'], axis=1)  # Features
y = data['sales']  # Target variable

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# Step 6: Making predictions
new_data = pd.DataFrame({
    'month': [4],
    'feature1': [350],
    'feature2': [30]
})
predicted_sales = model.predict(new_data)
print(f'Predicted Sales: {predicted_sales[0]:.2f}')
