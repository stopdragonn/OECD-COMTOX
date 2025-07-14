"""
Molecular Descriptors Data Loader
Scaling required - continuous features
"""

import openpyxl
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from ..preprocessing.smiles2fing import Smiles2Fing


def save_scaler(scaler, file_path):
    """Save scaler object to file"""
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(scaler, file_path)
    print(f"Scaler saved to {file_path}")


def load_data(file_path, fingerprint_type='MACCS', scaler_save_path=None):
    """
    Load data for Molecular Descriptors experiments (scaling required)
    
    Args:
        file_path (str): Path to Excel file
        fingerprint_type (str): Type of molecular fingerprint (used for naming)
        scaler_save_path (str, optional): Path to save the fitted scaler
        
    Returns:
        pd.DataFrame, pd.Series, StandardScaler: Scaled features, labels, and fitted scaler
    """
    df = pd.read_excel(file_path)

    # Generate fingerprints for consistency (though we may not use them for MD-only)
    drop_idx, fingerprints = Smiles2Fing(df.SMILES, fingerprint_type=fingerprint_type)

    # Extract molecular descriptors (exclude SMILES, ID columns, and target)
    exclude_cols = ['SMILES', 'Toxicity', 'Genotoxicity', 'DTXSID', 'Chemical', 'CasRN']
    descriptor_cols = [col for col in df.columns if col not in exclude_cols]
    
    descriptors = df[descriptor_cols].drop(index=drop_idx).reset_index(drop=True)

    # Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(descriptors)

    # Save scaler if path provided
    if scaler_save_path:
        save_scaler(scaler, scaler_save_path)

    # Handle labels
    label_col = None
    for col in ['Toxicity', 'Genotoxicity']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("No toxicity label column found. Expected 'Toxicity' or 'Genotoxicity'")

    if df[label_col].dtype == 'object':
        y = df[label_col].drop(drop_idx).replace({'negative': 0, 'positive': 1}).reset_index(drop=True)
    else:
        y = df[label_col].drop(drop_idx).reset_index(drop=True)

    return pd.DataFrame(scaled_features, columns=descriptors.columns), y, scaler
