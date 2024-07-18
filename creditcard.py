import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the dataset
file_path = 'creditcard.csv'  # Update with your actual path
data = pd.read_csv(file_path)

# Step 2: Data exploration
print(data.head())
print(data.isnull().sum())
print(data['Class'].value_counts())

# Step 3: Data preparation
X = data.drop(['Class'], axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Model evaluation
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 6: Making predictions
new_data = pd.DataFrame({
    'V1': [0.1],
    'V2': [-1.2],
    'V3': [0.5],
    'V4': [-0.5],
    'V5': [0.1],
    'V6': [-0.3],
    'V7': [0.2],
    'V8': [0.4],
    'V9': [0.3],
    'V10': [0.1],
    'V11': [-0.2],
    'V12': [0.0],
    'V13': [0.5],
    'V14': [-0.4],
    'V15': [0.6],
    'V16': [0.5],
    'V17': [-0.3],
    'V18': [0.2],
    'V19': [0.1],
    'V20': [-0.6],
    'V21': [0.4],
    'V22': [0.3],
    'V23': [0.2],
    'V24': [-0.1],
    'V25': [0.5],
    'V26': [0.4],
    'V27': [0.1],
    'V28': [-0.3],
    'Amount': [100]
})

predicted_class = model.predict(new_data)
print(f'Predicted Class: {predicted_class[0]}')
