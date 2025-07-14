"""
Molecular Fingerprints Data Loader
No scaling required - binary/discrete features
"""

import openpyxl
import pandas as pd
from ..preprocessing.smiles2fing import Smiles2Fing


def load_data(file_path, fingerprint_type='MACCS'):
    """
    Load data for Molecular Fingerprints experiments (no scaling required)
    
    Args:
        file_path (str): Path to Excel file
        fingerprint_type (str): Type of molecular fingerprint
        
    Returns:
        pd.DataFrame, pd.Series: Fingerprints and labels
    """
    df = pd.read_excel(file_path)
    
    drop_idx, fingerprints = Smiles2Fing(df.SMILES, fingerprint_type=fingerprint_type)
    
    # Handle different label column names and types
    label_col = None
    for col in ['Toxicity', 'Genotoxicity']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("No toxicity label column found. Expected 'Toxicity' or 'Genotoxicity'")
    
    # Data type handling
    if df[label_col].dtype == 'object':  # String type
        y = df[label_col].drop(drop_idx).replace({'negative': 0, 'positive': 1}).reset_index(drop=True)
    else:  # Already numeric
        y = df[label_col].drop(drop_idx).reset_index(drop=True)
          
    return fingerprints, y
