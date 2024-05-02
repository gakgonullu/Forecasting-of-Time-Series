import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from window_size_finder import find_best_window_size

# Generate synthetic time series data
time_series = np.sin(np.linspace(0, 20, 730)) + np.random.normal(0, 0.5, 730)

# Create pandas DataFrame
df = pd.DataFrame({'Value': time_series})
plt.plot(df)
# Define range of window sizes
min_window_size = 5
max_window_size = 25

# Find the best window size
best_window_size, best_mse = find_best_window_size(df, min_window_size, max_window_size)

print("Best Window Size:", best_window_size)
print("Corresponding MSE:", best_mse)

# Sliding window using DataFrame operations
X = []
y = []
for i in range(len(df) - best_window_size):
    X.append(df['Value'].iloc[i:i+best_window_size].values)
    y.append(df['Value'].iloc[i+best_window_size])

# Convert data to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate final MSE
final_mse = mean_squared_error(y_test, y_pred)
print("Final Mean Squared Error (MSE):", final_mse)

# Predict the next day
next_day_index = df.index[-1] + 1  

# Reshape the features for prediction
next_day_features = df['Value'].iloc[-best_window_size:].values.reshape(1, -1)

# Use the trained model to predict the value for the next day
next_day_prediction = rf_model.predict(next_day_features)

# Get the indices of the testing set in the original DataFrame
test_indices = df.index[-len(y_test):]

# Plot predicted and actual values against original time series indices
# Plot predicted and actual values against original time series indices
plt.figure(figsize=(10, 6))
plt.plot(test_indices, y_test, label='Actual')
plt.plot(test_indices, y_pred, label='Predicted')

# Plot the predicted value for the next day as a red dot
plt.plot(next_day_index, next_day_prediction, 'ro', label='Predicted (Next Day)', markersize=8)

plt.title("Actual and Predicted Values (Best Window Size)")
plt.xlabel("Sample Number")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
