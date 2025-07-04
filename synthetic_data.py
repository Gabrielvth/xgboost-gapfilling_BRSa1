# synthetic_data.py - simplified version without keras
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def make_synth_data(data, x_cols, y_col):
    """Create synthetic data using RandomForest instead of neural networks"""
    
    # Prepare data
    X = data[x_cols].dropna()
    y = data[y_col].dropna()
    
    # Align X and y
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    # Train a simple model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Generate synthetic data with some noise
    y_pred = model.predict(X_scaled)
    noise = np.random.normal(0, np.std(y) * 0.1, len(y_pred))
    y_synth = y_pred + noise
    
    # Create synthetic dataframe
    synth_data = data.copy()
    synth_data[y_col] = y_synth
    
    return synth_data
