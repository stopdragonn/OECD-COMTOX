#!/bin/bash

# Single experiment runner for molecular descriptors
# Usage: ./run_md_experiment.sh

echo "Running Molecular Descriptors experiment..."

# Configuration
MODEL="gbt"
FINGERPRINT="Morgan"  # Used for naming only
DATA_FILE="../data/raw/molecular_descriptors/TG201_Descriptor_desalt.xlsx"
OUTPUT_DIR="../results/models/molecular_descriptors/TG201"
SCALER_DIR="../results/models/molecular_descriptors/scalers"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCALER_DIR"

# Run experiment
python ../src/models/molecular_descriptors/${MODEL}.py \
    --fingerprint_type "$FINGERPRINT" \
    --file_path "$DATA_FILE" \
    --model_save_path "$OUTPUT_DIR" \
    --scaler_save_path "${SCALER_DIR}/scaler_${FINGERPRINT}_${MODEL}.joblib"

echo "Molecular Descriptors experiment completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Scaler saved to: $SCALER_DIR"
