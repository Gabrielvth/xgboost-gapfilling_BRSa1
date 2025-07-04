# ameriflux_transform.py

import pandas as pd
import numpy as np
import os

def transform_ameriflux_data(filepath):
    print(f"Reading file: {filepath}")
    
    df = pd.read_csv(
        filepath,
        sep=',',
        comment='#',
        skip_blank_lines=True,
        encoding='utf-8'
    )
    
    print(f"Successfully read {len(df)} rows and {len(df.columns)} columns")
    
    # Map column names
    column_mapping = {
        'NEE_PI': 'NEE',
        'FC': 'FC'
    }
    df = df.rename(columns=column_mapping)
    
    # Convert timestamp to datetime and set as index
    if 'TIMESTAMP_START' in df.columns:
        # Convert YYYYMMDDHHMM format to datetime
        df['datetime'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
        df.set_index('datetime', inplace=True)
        print("Successfully converted timestamp to datetime index")
    else:
        raise ValueError("TIMESTAMP_START column not found")
    
    print(f"Available columns after mapping: {df.columns.tolist()}")
    
    return df

def validate_data(df):
    """Validate transformed data meets requirements"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be datetime")
    if df.isnull().all().any():
        print("Warning: Some columns contain all NaN values")
    return True