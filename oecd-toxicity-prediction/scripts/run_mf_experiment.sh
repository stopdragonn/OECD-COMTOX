#!/bin/bash

# Single experiment runner for molecular fingerprints
# Usage: ./run_mf_experiment.sh

echo "Running Molecular Fingerprints experiment..."

# Configuration
MODEL="gbt"
FINGERPRINT="Morgan"
DATA_FILE="../data/raw/molecular_fingerprints/TG201.xlsx"
OUTPUT_DIR="../results/models/molecular_fingerprints/TG201"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run experiment
python ../src/models/molecular_fingerprints/${MODEL}.py \
    --fingerprint_type "$FINGERPRINT" \
    --file_path "$DATA_FILE" \
    --model_save_path "$OUTPUT_DIR"

echo "Molecular Fingerprints experiment completed!"
echo "Results saved to: $OUTPUT_DIR"
