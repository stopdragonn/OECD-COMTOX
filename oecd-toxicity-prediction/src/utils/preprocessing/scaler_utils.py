"""
Scaler utilities for molecular descriptor preprocessing
"""

import joblib
from sklearn.preprocessing import StandardScaler


def save_scaler(scaler, file_path):
    """
    Save scaler object to file
    
    Args:
        scaler (StandardScaler): Fitted scaler object
        file_path (str): Path to save the scaler
    """
    joblib.dump(scaler, file_path)
    print(f"Scaler saved to {file_path}")


def load_scaler(file_path):
    """
    Load scaler object from file
    
    Args:
        file_path (str): Path to the saved scaler
        
    Returns:
        StandardScaler: Loaded scaler object
    """
    return joblib.load(file_path)


def apply_scaling(data, scaler=None, fit=True):
    """
    Apply scaling to data
    
    Args:
        data: Data to scale
        scaler (StandardScaler, optional): Existing scaler to use
        fit (bool): Whether to fit the scaler or just transform
        
    Returns:
        tuple: (scaled_data, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    
    return scaled_data, scaler
