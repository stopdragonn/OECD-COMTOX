import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import argparse
from .read_data import load_data

def main(model_path, fingerprint_type, file_path, output_dir):
    """
    Main function to perform SHAP analysis on the trained model.
    
    Args:
        model_path (str): Path to the trained model file
        fingerprint_type (str): Type of fingerprint used for the model
        file_path (str): Path to the data file
        output_dir (str): Directory to save the output plots
    """
    print(f"Starting SHAP analysis for model: {model_path}")
    print(f"Using fingerprint type: {fingerprint_type}")
    print(f"Loading data from: {file_path}")
    
    # Extract the target name from the file path for naming outputs
    import os
    target_name = os.path.basename(file_path).split('.')[0]
    
    # Set output file paths
    output_bar = f"{output_dir}/{target_name}_{fingerprint_type}_shap_top10_bar.png"
    output_beeswarm = f"{output_dir}/{target_name}_{fingerprint_type}_shap_beeswarm.png"
    
    print("Loading data...")
    # Load data using the existing function
    X, y = load_data(file_path=file_path, fingerprint_type=fingerprint_type)
    
    print("Loading model...")
    # Load the trained model
    model = joblib.load(model_path)
    
    print("Creating SHAP explainer...")
    # Create a SHAP TreeExplainer for the model
    explainer = shap.TreeExplainer(model)
    
    print("Calculating SHAP values...")
    # Get SHAP values for positive class (class index 1)
    shap_vals = explainer.shap_values(X)
    
    # For binary classification models, shap_values returns a list with two elements
    # If the model is RandomForestClassifier, the first element is for negative class and second for positive class
    if isinstance(shap_vals, list) and len(shap_vals) > 1:
        shap_vals = shap_vals[1]  # Use positive class for binary classification
    
    print("Generating bar chart for top 10 features...")
    # Generate and save bar chart showing mean absolute SHAP values
    # for the top 10 most important features
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, plot_type='bar', max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(output_bar, bbox_inches='tight', dpi=300)
    plt.clf()
    print(f"Bar chart saved to {output_bar}")
    
    print("Generating beeswarm plot...")
    # Generate and save beeswarm plot showing SHAP values distribution
    # for each feature and their impact on model output
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_vals, X, show=False)
    plt.tight_layout()
    plt.savefig(output_beeswarm, bbox_inches='tight', dpi=300)
    plt.clf()
    print(f"Beeswarm plot saved to {output_beeswarm}")
    
    # Calculate the mean absolute SHAP value for each feature
    print("\nTop 10 most important features based on mean |SHAP| value:")
    mean_shap = np.abs(shap_vals).mean(axis=0)
    
    # Get feature names and their importance
    feature_names = X.columns.tolist()
    feature_importance = list(zip(feature_names, mean_shap))
    
    # Sort by importance (descending)
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10 features
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        print(f"{i}. {feature}: {importance:.6f}")
    
    # Save feature importance to CSV file
    importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'SHAP_Importance'])
    importance_csv = f"{output_dir}/{target_name}_{fingerprint_type}_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    print(f"Feature importance saved to {importance_csv}")
    
    print("\nSHAP analysis complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform SHAP analysis on a trained model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model file')
    parser.add_argument('--fingerprint_type', type=str, default='MACCS',
                        help='Type of fingerprint used for the model')
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to the data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output plots')
                        
    args = parser.parse_args()
    
    main(args.model_path, args.fingerprint_type, args.file_path, args.output_dir)
