#!/usr/bin/env python3
"""
OECD Toxicity Prediction - Pre-trained Model Usage Script

This script demonstrates how to use pre-trained models for toxicity prediction.
It supports Molecular Fingerprints (MF), Molecular Descriptors (MD), and Combined (MF+MD) models.

Usage Examples:
    # Molecular Fingerprints (no scaler needed)
    python predict_toxicity.py --model_type MF --fingerprint_type RDKit \
        --input_file data.xlsx \
        --model_path ../Paper/Model/Geno_model/TG471_best_model_RDKit_lr.joblib
    
    # Molecular Descriptors (scaler required)
    python predict_toxicity.py --model_type MD --fingerprint_type RDKit \
        --input_file data.xlsx \
        --model_path ../Paper/Model/Eco_model/TG202_best_model_RDKit_xgb.joblib \
        --scaler_path ../Paper/Model/Eco_model/TG202_scaler_RDKit_xgb.joblib

Pre-trained models are located at: /home2/jjy0605/Toxicity/0817_Genotoxicity/OECD-COMTOX/Paper/Model/

Author: OECD Toxicity Prediction Team
"""

import argparse
import sys
import os
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from utils.smiles2fing import Smiles2Fing
    from utils.read_data import load_data
    from sklearn.preprocessing import StandardScaler
    from rdkit import RDLogger, Chem
    RDLogger.DisableLog('rdApp.*')
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you have installed all required packages and are in the correct directory.")
    sys.exit(1)

def validate_smiles(smiles_list):
    """Validate SMILES strings using RDKit"""
    valid_indices = []
    invalid_smiles = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_indices.append(i)
        else:
            invalid_smiles.append((i, smiles))
    
    return valid_indices, invalid_smiles

def load_and_prepare_data(input_file, fingerprint_type, model_type):
    """Load and prepare data based on model type"""
    
    # Read input Excel file
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")
    
    # Check required columns
    if 'SMILES' not in df.columns:
        raise ValueError("Input file must contain 'SMILES' column")
    
    # Validate SMILES
    valid_indices, invalid_smiles = validate_smiles(df['SMILES'].tolist())
    
    if invalid_smiles:
        print(f"Warning: Found {len(invalid_smiles)} invalid SMILES:")
        for idx, smiles in invalid_smiles[:5]:  # Show first 5
            print(f"  Row {idx}: {smiles}")
        if len(invalid_smiles) > 5:
            print(f"  ... and {len(invalid_smiles) - 5} more")
    
    # Filter valid data
    df_valid = df.iloc[valid_indices].copy()
    
    if model_type == 'MF':
        # Generate molecular fingerprints using the Smiles2Fing function
        ms_none_idx, fingerprints_df = Smiles2Fing(df_valid['SMILES'].tolist(), fingerprint_type)
        X = fingerprints_df.values
        return X, df_valid, valid_indices
    
    elif model_type in ['MD', 'COMBINED']:
        # For MD and Combined models, descriptors should be pre-computed in the input file
        descriptor_cols = [col for col in df_valid.columns if col not in ['SMILES', 'Toxicity', 'Genotoxicity']]
        
        if not descriptor_cols:
            raise ValueError(f"For {model_type} models, input file must contain molecular descriptor columns")
        
        if model_type == 'MD':
            X = df_valid[descriptor_cols].values
        else:  # COMBINED
            # Generate fingerprints and concatenate with descriptors
            ms_none_idx, fingerprints_df = Smiles2Fing(df_valid['SMILES'].tolist(), fingerprint_type)
            fingerprints = fingerprints_df.values
            descriptors = df_valid[descriptor_cols].values
            X = np.concatenate([fingerprints, descriptors], axis=1)
        
        return X, df_valid, valid_indices, descriptor_cols
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def predict_toxicity(args):
    """Main prediction function"""
    
    print(f"OECD Toxicity Prediction - Using {args.model_type} Model")
    print(f"Fingerprint Type: {args.fingerprint_type}")
    print(f"Input File: {args.input_file}")
    print(f"Model Path: {args.model_path}")
    print("-" * 50)
    
    # Load model
    try:
        model = joblib.load(args.model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")
    
    # Load scaler if needed
    scaler = None
    if args.model_type in ['MD', 'COMBINED'] and args.scaler_path:
        try:
            scaler = joblib.load(args.scaler_path)
            print(f"✓ Scaler loaded successfully")
        except Exception as e:
            raise ValueError(f"Error loading scaler: {e}")
    elif args.model_type in ['MD', 'COMBINED'] and not args.scaler_path:
        raise ValueError(f"Scaler path is required for {args.model_type} models")
    
    # Load and prepare data
    print("Loading and preparing data...")
    if args.model_type == 'MF':
        X, df_valid, valid_indices = load_and_prepare_data(args.input_file, args.fingerprint_type, args.model_type)
    else:
        X, df_valid, valid_indices, descriptor_cols = load_and_prepare_data(args.input_file, args.fingerprint_type, args.model_type)
        print(f"✓ Found {len(descriptor_cols)} descriptor columns")
    
    print(f"✓ Data prepared: {X.shape[0]} valid compounds, {X.shape[1]} features")
    
    # Apply scaling if needed
    if scaler is not None:
        if args.model_type == 'MD':
            X = scaler.transform(X)
        elif args.model_type == 'COMBINED':
            # Only scale the descriptor part (not fingerprints)
            _, fp_df = Smiles2Fing(['CCO'], args.fingerprint_type)
            n_fingerprints = fp_df.shape[1]
            X_descriptors = X[:, n_fingerprints:]
            X_descriptors_scaled = scaler.transform(X_descriptors)
            X = np.concatenate([X[:, :n_fingerprints], X_descriptors_scaled], axis=1)
        print("✓ Data scaled using pre-trained scaler")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results DataFrame
    results = df_valid.copy()
    results['Prediction'] = predictions
    results['Probability_Negative'] = probabilities[:, 0]
    results['Probability_Positive'] = probabilities[:, 1]
    results['Confidence'] = np.max(probabilities, axis=1)
    
    # Add interpretation
    results['Interpretation'] = results['Prediction'].map({0: 'Non-toxic', 1: 'Toxic'})
    
    # Save results
    output_file = args.output_file or args.input_file.replace('.xlsx', '_predictions.xlsx')
    results.to_excel(output_file, index=False)
    
    print(f"✓ Predictions saved to: {output_file}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total compounds processed: {len(results)}")
    print(f"Predicted toxic: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
    print(f"Predicted non-toxic: {len(predictions) - sum(predictions)} ({(len(predictions) - sum(predictions))/len(predictions)*100:.1f}%)")
    print(f"Average confidence: {np.mean(results['Confidence']):.3f}")
    
    # If true labels are available, calculate metrics
    if 'Toxicity' in df_valid.columns or 'Genotoxicity' in df_valid.columns:
        true_col = 'Toxicity' if 'Toxicity' in df_valid.columns else 'Genotoxicity'
        y_true = df_valid[true_col].values
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        auc = roc_auc_score(y_true, probabilities[:, 1])
        
        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"ROC AUC: {auc:.3f}")

def main():
    parser = argparse.ArgumentParser(description='OECD Toxicity Prediction using Pre-trained Models')
    
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['MF', 'MD', 'COMBINED'],
                       help='Type of model: MF (Molecular Fingerprints), MD (Molecular Descriptors), COMBINED (MF+MD)')
    
    parser.add_argument('--fingerprint_type', type=str, required=True,
                       choices=['MACCS', 'Morgan', 'RDKit', 'Layered', 'Pattern'],
                       help='Type of molecular fingerprint')
    
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input Excel file containing SMILES and optionally descriptors')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pre-trained model (.joblib file)')
    
    parser.add_argument('--scaler_path', type=str, default=None,
                       help='Path to scaler file (.joblib file) - required for MD and COMBINED models')
    
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path (default: input_file_predictions.xlsx)')
    
    args = parser.parse_args()
    
    try:
        predict_toxicity(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
