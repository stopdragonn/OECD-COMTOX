"""
Combined Features (MF+MD) Data Loader
Scaling required for descriptor portion
"""

import openpyxl
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from ..preprocessing.smiles2fing import Smiles2Fing


def save_scaler(scaler, file_path):
    """Save scaler object to file"""
    joblib.dump(scaler, file_path)
    print(f"Scaler saved to {file_path}")


def load_data(file_path, fingerprint_type='MACCS', scaler_save_path=None):
    """
    Load data for Combined (MF+MD) experiments (scaling required for descriptors)
    
    Args:
        file_path (str): Path to Excel file
        fingerprint_type (str): Type of molecular fingerprint
        scaler_save_path (str, optional): Path to save the fitted scaler
        
    Returns:
        pd.DataFrame, pd.Series, StandardScaler: Combined features, labels, and fitted scaler
    """
    df = pd.read_excel(file_path)

    # Generate molecular fingerprints
    drop_idx, fingerprints = Smiles2Fing(df.SMILES, fingerprint_type=fingerprint_type)

    # Extract molecular descriptors (exclude SMILES, ID columns, and target)
    exclude_cols = ['SMILES', 'Toxicity', 'Genotoxicity', 'DTXSID', 'Chemical', 'CasRN']
    descriptor_cols = [col for col in df.columns if col not in exclude_cols]
    
    descriptors = df[descriptor_cols].drop(index=drop_idx).reset_index(drop=True)

    # Combine fingerprints and descriptors
    combined_features = pd.concat([fingerprints, descriptors], axis=1)

    # Scale only the descriptor portion
    scaler = StandardScaler()
    scaled_descriptors = scaler.fit_transform(descriptors)
    
    # Combine scaled descriptors with unscaled fingerprints
    final_features = pd.concat([
        fingerprints,
        pd.DataFrame(scaled_descriptors, columns=descriptors.columns)
    ], axis=1)

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

    return final_features, y, scaler
