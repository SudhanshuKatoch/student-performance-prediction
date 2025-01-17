import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os

# Load dataset
dataset_path = r"C:\student-performance-prediction\data\StudentPerformanceFactors.csv"
data = pd.read_csv(dataset_path)

# Define target column
target_column = 'Exam_Score'

if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in the dataset. Check your dataset.")

# Preprocess dataset
X = data.drop(columns=[target_column])  # Features
y = data[target_column]  # Target

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
os.makedirs("models", exist_ok=True)
scaler_path = r"C:\student-performance-prediction\models\scaler.pkl"
joblib.dump(scaler, scaler_path)

# Build neural network model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')  # For regression
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Save the trained model
model_path = r"C:\student-performance-prediction\models\nn_model.h5"
model.save(model_path)

# Evaluate the model
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test MAE: {mae}")
print(f"Model and scaler have been saved successfully!")
