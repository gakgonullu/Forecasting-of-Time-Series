import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def find_best_window_size(df, min_window_size, max_window_size):
    mse_values = {}
    
    for window_size in range(min_window_size, max_window_size + 1):
        X = []
        y = []
        for i in range(len(df) - window_size):
            X.append(df['Value'].iloc[i:i+window_size].values)
            y.append(df['Value'].iloc[i+window_size])

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        
        mse_values[window_size] = mse

    best_window_size = min(mse_values, key=mse_values.get)
    best_mse = mse_values[best_window_size]
    
    return best_window_size, best_mse
