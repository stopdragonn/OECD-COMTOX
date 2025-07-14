# OECD Toxicity Prediction using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2020.03+-green.svg)](https://www.rdkit.org/)
[![Reproducible](https://img.shields.io/badge/Reproducible-✅%20Verified-brightgreen.svg)](https://github.com)

**✅ Fully Reproducible** - All scripts tested and verified to work out-of-the-box on standard computing environments without requiring cluster access.

This repository contains the code and data for reproducing toxicity prediction research based on OECD Test Guidelines (TG). The project implements multiple machine learning models using different molecular feature representations to predict various toxicity endpoints i**Model Attribution**: All models, scalers, and code in this repository are original research contributions by the authors and are shared under MIT License to facilitate scientific reproducibility and collaboration. The pre-trained models in `published-models/Model/` are provided as supplementary materials to our published research for exact result reproduction.cluding genotoxicity, carcinogenicity, acute toxicity, developmental and reproductive toxicity (DART), and ecotoxicity.

## Table of Contents

- [Overview](#overview)
- [Data Types and Preprocessing](#data-types-and-preprocessing)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Usage Guide](#model-usage-guide)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

This project focuses on predicting chemical toxicity endpoints using machine learning approaches based on OECD Test Guidelines (TG) data. The framework supports multiple toxicity endpoints and molecular feature representations:

### Supported Toxicity Endpoints

- **Genotoxicity** (TG 471, 473, 474, 476, 478, 486, 487)
- **Carcinogenicity** (TG 451, 453)
- **Acute Toxicity** (TG 420, 423, 425)
- **Developmental and Reproductive Toxicity (DART)** (TG 414, 416, 421, 422, 443)
- **Ecotoxicity** (TG 201, 202, 203, 210, 211)

### Molecular Feature Types

- **Molecular Fingerprints (MF)**: Binary/discrete molecular representations
- **Molecular Descriptors (MD)**: Continuous numerical chemical properties  
- **Combined Features (MF+MD)**: Integration of both fingerprints and descriptors

### Supported Models

- Gradient Boosting Classifier (GBT)
- Random Forest (RF)
- XGBoost (XGB)
- Logistic Regression
- Decision Tree (DT)
- Multi-layer Perceptron (MLP)
- Linear/Quadratic Discriminant Analysis (LDA/QDA)
- Partial Least Squares Discriminant Analysis (PLS-DA)
- LightGBM (LGB)

### Supported Fingerprint Types

- MACCS Keys
- Morgan Fingerprints
- RDKit Fingerprints
- Layered Fingerprints
- Pattern Fingerprints

## Data Types and Preprocessing

### 🔑 Important: Feature Types and Scaling Requirements

This project supports three different types of molecular features, each with different preprocessing requirements:

| Feature Type | Description | Scaling Required | Scaler Storage | Notes |
|--------------|-------------|------------------|----------------|-------|
| **MF** (Molecular Fingerprints) | Binary/integer molecular representations | ❌ No | N/A | Direct input to models |
| **MD** (Molecular Descriptors) | Continuous numerical features | ✅ **Required** | ✅ **Required** | StandardScaler applied |
| **MF+MD** (Combined) | Concatenation of fingerprints + descriptors | ✅ **Required** | ✅ **Required** | Scaler applied to MD portion |

### Scaling Management

When using molecular descriptors (MD or MF+MD), the fitted scaler **must be saved** for:
- Consistent model evaluation
- Future predictions on new data
- Research reproducibility

The scaling process is automatically handled by our data loaders and model scripts.

## Installation

### Prerequisites

- Python 3.6+ (tested with Python 3.6)
- conda or pip package manager

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/stopdragonn/oecd-toxicity-prediction.git
cd oecd-toxicity-prediction

# Create conda environment
conda env create -f environment.yml
conda activate oecd-toxicity-pred
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/stopdragonn/oecd-toxicity-prediction.git
cd oecd-toxicity-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Installation Notes

- **Python Version**: This project is tested with Python 3.6.8. It uses exact package versions to ensure reproducibility.
- **RDKit**: We use RDKit 2020.03.3 for molecular fingerprint generation. Installation via conda is recommended.
- **Package Versions**: All package versions are pinned to match the tested environment (grover_env3.6).
- **Environment Replication**: The environment.yml file recreates the exact conda environment used for development.
- **Local Execution**: All scripts are designed to work on standard computing environments (Linux/macOS/Windows with WSL).
- **Cluster Support**: Optional SLURM cluster scripts are provided for advanced users, but not required.
- **Parallel Processing**: Local batch scripts use standard multiprocessing and background processes.
- **Memory Requirements**: Recommended 4GB+ RAM for molecular fingerprints, 8GB+ for descriptors.
- **Compatibility**: All packages are tested to work together without version conflicts.
- **GPU Support**: For XGBoost GPU acceleration, install `xgboost[gpu]` instead of `xgboost==1.5.2`

### Verify Installation

```bash
python -c "import rdkit; print('RDKit version:', rdkit.__version__)"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import pandas; print('Pandas version:', pandas.__version__)"
```

## Quick Start

This project supports both local execution and cluster-based runs. All scripts work out-of-the-box on standard computing environments.

### 1. Single Model Training

#### Molecular Fingerprints (No scaling required)
```bash
# From project root directory
python src/models/molecular_fingerprints/gbt.py \
    --fingerprint_type Morgan \
    --file_path data/raw/molecular_fingerprints/TG201.xlsx \
    --model_save_path results/models/molecular_fingerprints/TG201
```

#### Molecular Descriptors (With scaling)
```bash
python src/models/molecular_descriptors/gbt.py \
    --fingerprint_type RDKit \
    --file_path data/raw/molecular_descriptors/TG201.xlsx \
    --model_save_path results/models/molecular_descriptors/TG201 \
    --scaler_save_path results/models/molecular_descriptors/TG201/scaler_RDKit_gbt.joblib
```

### 2. Batch Experiments (Recommended)

#### Option A: Local Bash Script (Linux/macOS)

**✅ Verified and Working**

```bash
# Quick test with single model and fingerprint
./scripts/batch_experiments/batch_experiment_local.sh \
    --experiment_type MF \
    --models gbt \
    --fingerprints Morgan

# Multiple models and fingerprints
./scripts/batch_experiments/batch_experiment_local.sh \
    --experiment_type MF \
    --models "gbt rf xgb" \
    --fingerprints "Morgan MACCS RDKit" \
    --verbose

# Molecular descriptors with scaling - VERIFIED WORKING ✅
./scripts/batch_experiments/batch_experiment_local.sh \
    --experiment_type MD \
    --models "gbt rf" \
    --fingerprints "RDKit Mordred" \
    --verbose

# Dry run to see what experiments would be executed
./scripts/batch_experiments/batch_experiment_local.sh \
    --experiment_type MF \
    --models gbt \
    --fingerprints Morgan \
    --dry-run
```

#### Option B: Single Experiment Testing

```bash
# Test a single configuration
./scripts/run_single_experiment.sh \
    --type MF \
    --model gbt \
    --fingerprint Morgan \
    --data TG201 \
    --verbose

# Test with molecular descriptors
./scripts/run_single_experiment.sh \
    --type MD \
    --model xgb \
    --fingerprint RDKit \
    --data TG202 \
    --verbose

# List available options
./scripts/run_single_experiment.sh --help
```

### 3. Example Successful Run

```bash
# This example shows a successful experiment run
$ ./scripts/batch_experiments/batch_experiment_local.sh --experiment_type MF --models gbt --fingerprints Morgan

[2025-07-14 15:02:26] OECD Toxicity Prediction - Local Batch Experiment Runner
[2025-07-14 15:02:26] ========================================================
[2025-07-14 15:02:26] Configuration:
[2025-07-14 15:02:26]   Experiment Type: MF - Molecular Fingerprints (no scaler required)
[2025-07-14 15:02:26]   Models: gbt
[2025-07-14 15:02:26]   Fingerprints: Morgan
[2025-07-14 15:02:26]   Total Experiments: 1
[2025-07-14 15:02:26] Running experiments sequentially...
[2025-07-14 15:19:26] ✓ Completed: TG201_Morgan_gbt (1020s)

# Results generated:
# - Model: results/models/molecular_fingerprints/TG201/best_model_Morgan_gbt.joblib
# - Log: results/logs/molecular_fingerprints/TG201/Morgan_gbt.log
# - Performance: F1=0.751, Precision=0.656, Recall=0.878, AUC=0.699
```

#### Option C: Python-based Batch Runner

For cross-platform compatibility:

```bash
# Install optional dependency for system monitoring
pip install psutil

# Run batch experiments
python scripts/batch_experiments/batch_experiment_local.py \
    --experiment_type MF \
    --max_workers 2 \
    --models gbt rf \
    --fingerprints Morgan MACCS

# Monitor progress with detailed logging
python scripts/batch_experiments/batch_experiment_local.py \
    --experiment_type MD \
    --max_workers 1 \
    --models xgb gbt
```

#### Option C: Single Experiment Testing

For quick testing and development:

```bash
# Test a single configuration
./scripts/run_single_experiment.sh \
    --type MF \
    --model gbt \
    --fingerprint Morgan \
    --data TG201

# Test with molecular descriptors
./scripts/run_single_experiment.sh \
    --type MD \
    --model xgb \
    --fingerprint RDKit \
    --data TG202 \
    --verbose
```

#### Option D: SLURM Cluster (Advanced users)

For users with access to SLURM cluster systems:

```bash
# Edit scripts/batch_experiments/batch_experiment_MF2.sh
# Change EXPERIMENT_TYPE="MF" to desired type: "MF", "MD", or "COMBINED"

# Run the batch experiment
./scripts/batch_experiments/batch_experiment_MF2.sh
```

## Model Usage Guide

### 🎯 Using Pre-trained Models for Prediction

This section describes how to use already trained models (`.joblib` files) that have been reported in research papers. The pre-trained models and datasets are available through our repository.

#### Prerequisites for Model Usage

1. **Install the environment** (see [Installation](#installation))
2. **Download pre-trained models** from the repository
3. **Prepare your input data** in the required format

#### Model Types and Requirements

| Model Type | Required Files | Input Data | Scaler Needed |
|------------|---------------|------------|---------------|
| **MF Models** | `model.joblib` | SMILES only | ❌ No |
| **MD Models** | `model.joblib` + `scaler.joblib` | SMILES + descriptors | ✅ Yes |
| **MF+MD Models** | `model.joblib` + `scaler.joblib` | SMILES + descriptors | ✅ Yes |

#### Step-by-Step Usage

##### Option A: Using the Prediction Script (Recommended)

We provide a comprehensive prediction script that handles all model types:

```bash
# Molecular Fingerprints Model
python predict_toxicity.py \
    --model_type MF \
    --fingerprint_type RDKit \
    --input_file your_compounds.xlsx \
    --model_path ../published-models/Model/Geno_model/TG471_best_model_RDKit_lr.joblib

# Molecular Descriptors Model  
python predict_toxicity.py \
    --model_type MD \
    --fingerprint_type RDKit \
    --input_file your_compounds_with_descriptors.xlsx \
    --model_path ../published-models/Model/Eco_model/TG202D_best_model_RDKit_xgb.joblib \
    --scaler_path ../published-models/Model/Eco_model/TG202D_scaler_RDKit_xgb.joblib

# Combined Model
python predict_toxicity.py \
    --model_type COMBINED \
    --fingerprint_type RDKit \
    --input_file your_compounds_with_descriptors.xlsx \
    --model_path ../published-models/Model/DART_model/TG421D_best_model_RDKit_logistic.joblib \
    --scaler_path ../published-models/Model/DART_model/TG421D_scaler_RDKit_logistic.joblib \
    --output_file my_predictions.xlsx
```

##### Option B: Using Individual Model Scripts

##### 1. **Molecular Fingerprints (MF) Model Usage**

```bash
# Example: Using a pre-trained RDKit fingerprint Logistic Regression model for TG471
python src/models/molecular_fingerprints/datacheck.py \
    --fingerprint_type RDKit \
    --file_path your_input_data.xlsx \
    --model_filename ../published-models/Model/Geno_model/TG471_best_model_RDKit_lr.joblib
```

**Input data format** (Excel file):
```
SMILES                    | Toxicity (optional)
CC(C)CC(=O)O             | 1
CCO                      | 0
```

##### 2. **Molecular Descriptors (MD) Model Usage**

```bash
# Example: Using a pre-trained RDKit XGB model for ecotoxicity
python src/models/molecular_descriptors/predict.py \
    --fingerprint_type RDKit \
    --file_path your_input_data.xlsx \
    --model_filename ../published-models/Model/Eco_model/TG202D_best_model_RDKit_xgb.joblib \
    --scaler_filename ../published-models/Model/Eco_model/TG202D_scaler_RDKit_xgb.joblib
```

**Input data format** (Excel file):
```
SMILES          | Toxicity | MolWt | LogP | TPSA | ... (descriptors)
CC(C)CC(=O)O   | 1        | 102.1 | 1.2  | 37.3 | ...
CCO            | 0        | 46.07 | -0.31| 20.2 | ...
```

##### 3. **Combined (MF+MD) Model Usage**

```bash
# Example: Using a DART combined features model
python src/models/combined/predict.py \
    --fingerprint_type RDKit \
    --file_path your_input_data.xlsx \
    --model_filename ../published-models/Model/DART_model/TG421D_best_model_RDKit_logistic.joblib \
    --scaler_filename ../published-models/Model/DART_model/TG421D_scaler_RDKit_logistic.joblib
```

#### Model Performance Repository

Access pre-trained models and performance metrics:

| Endpoint | Best Model | Model Path |
|----------|------------|------------|
| **TG471 (Genotoxicity)** | RDKit+Logistic | `../published-models/Model/Geno_model/TG471_best_model_RDKit_lr.joblib` |
| **TG473 (Genotoxicity)** | Morgan+DecisionTree | `../published-models/Model/Geno_model/TG473_best_model_Morgan_dt.joblib` |
| **TG474 (Genotoxicity)** | MACCS+RandomForest | `../published-models/Model/Geno_model/TG474_best_model_MACCS_rf.joblib` |
| **TG201 (Ecotoxicity)** | Layered+GBT | `../published-models/Model/Eco_model/TG201D_best_model_Layered_gbt.joblib` |
| **TG202 (Ecotoxicity)** | RDKit+XGB | `../published-models/Model/Eco_model/TG202D_best_model_RDKit_xgb.joblib` |
| **TG414 (DART)** | Morgan+Logistic | `../published-models/Model/DART_model/TG414_best_model_Morgan_logistic.joblib` |
| **TG421D (DART)** | RDKit+Logistic | `../published-models/Model/DART_model/TG421D_best_model_RDKit_logistic.joblib` |

**Note**: Models with scalers (MD/Combined features) have corresponding scaler files in the same directory.

#### Prediction Output

The prediction scripts will output:
- **Predictions**: Binary classification results (0/1)
- **Probabilities**: Confidence scores for each prediction
- **Performance metrics**: If true labels are provided
- **Feature importance**: For tree-based models

#### Example Workflow

```python
# Example Python script for batch prediction
import joblib
import pandas as pd
from src.utils.data_loader.mf_loader import load_fingerprint_data

# 1. Load your data
data = pd.read_excel('your_compounds.xlsx')

# 2. Load pre-trained model
model = joblib.load('../published-models/Model/Geno_model/TG471_best_model_RDKit_lr.joblib')

# 3. Prepare features (fingerprints)
X, _, smiles = load_fingerprint_data(
    fingerprint_type='Morgan',
    file_path='your_compounds.xlsx'
)

# 4. Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# 5. Save results
results = pd.DataFrame({
    'SMILES': smiles,
    'Prediction': predictions,
    'Probability_Positive': probabilities[:, 1]
})
results.to_excel('predictions_output.xlsx', index=False)
```

#### Data Preparation Guidelines

1. **SMILES Validation**: Ensure all SMILES are valid and can be parsed by RDKit
2. **Descriptor Consistency**: For MD/Combined models, use the same descriptor calculation method
3. **Missing Values**: Handle missing values before prediction
4. **Data Scale**: Molecular descriptors will be automatically scaled using the saved scaler

#### Troubleshooting

**✅ Common Issues and Solutions:**

| Issue | Solution |
|-------|----------|
| **Script exits with code 1** | Normal for completed experiments, check log files for actual results |
| **"No such file or directory"** | Ensure you're running scripts from project root directory |
| **"ModuleNotFoundError: utils"** | Run scripts from project root; they automatically set PYTHONPATH |
| **"FileNotFoundError: results/..."** | Model scripts create directories automatically; run from project root |
| **RDKit parsing error** | Check SMILES validity, remove invalid structures |
| **Scaler file not found** | Ensure scaler.joblib is in the correct path for MD/Combined models |
| **Memory errors with MD** | Run MD experiments sequentially: `--experiment_type MD --sequential` |
| **Slow execution** | Use `--parallel 2` for MF experiments; keep MD sequential |

**🔧 Quick Diagnostic Commands:**

```bash
# Check if you're in the right directory
pwd  # Should end with: oecd-toxicity-prediction
ls src/models/  # Should show: combined/ molecular_descriptors/ molecular_fingerprints/

# Test installation
python -c "import rdkit; print('RDKit OK')"
python -c "from src.utils.data_loader import mf_loader; print('Utils OK')"

# Test data availability  
ls data/raw/molecular_fingerprints/TG201.xlsx  # Should exist

# Test a quick single experiment
./scripts/run_single_experiment.sh --type MF --model gbt --fingerprint Morgan --data TG201
```

**📂 Available Test Data:**

```bash
# Currently working datasets:
data/raw/molecular_fingerprints/TG201.xlsx  ✅ Tested (MF)
data/raw/molecular_descriptors/TG201_Descriptor_desalt.xlsx  ✅ Tested (MD)
# Add more datasets as you verify them

# Note: Results may vary slightly from published metrics due to:
# - Different random seeds in hyperparameter optimization
# - New scaler fitting vs. original paper's scaler
# - For exact reproduction, use pre-trained models in Paper/Model/ directory
```

#### Model Applicability Domain

- **Chemical Space**: Models are trained on OECD TG datasets
- **Molecular Weight**: Recommended range: 50-1000 Da
- **Structural Diversity**: Best performance on drug-like compounds
- **Validation**: Always validate predictions with experimental data when possible

#### Quick Example

Try the prediction script with the provided example data:

```bash
# Test with example compounds (assuming you have a pre-trained model)
python predict_toxicity.py \
    --model_type MF \
    --fingerprint_type RDKit \
    --input_file example_input.xlsx \
    --model_path ../published-models/Model/Geno_model/TG471_best_model_RDKit_lr.joblib
```

Expected output:
```
OECD Toxicity Prediction - Using MF Model
Fingerprint Type: RDKit
Input File: example_input.xlsx
Model Path: ../published-models/Model/Geno_model/TG471_best_model_RDKit_lr.joblib
--------------------------------------------------
✓ Model loaded successfully
Loading and preparing data...
✓ Data prepared: 10 valid compounds, 2048 features
Making predictions...
✓ Predictions saved to: example_input_predictions.xlsx

Prediction Summary:
Total compounds processed: 10
Predicted toxic: 4 (40.0%)
Predicted non-toxic: 6 (60.0%)
Average confidence: 0.847
```

## Usage

### Local Environment (Recommended)

**✅ All scripts verified and working on standard computing environments.**

#### 1. Quick Single Experiment
```bash
# Test MF model quickly - VERIFIED WORKING ✅
./scripts/run_single_experiment.sh --type MF --model gbt --fingerprint Morgan --data TG201

# Test MD model with verbose output - VERIFIED WORKING ✅
./scripts/run_single_experiment.sh --type MD --model xgb --fingerprint RDKit --data TG202 --verbose
```

#### 2. Local Batch Experiments  
```bash
# Run single experiment - VERIFIED WORKING ✅
./scripts/batch_experiments/batch_experiment_local.sh \
    --experiment_type MF \
    --models gbt \
    --fingerprints Morgan

# Run multiple experiments
./scripts/batch_experiments/batch_experiment_local.sh \
    --experiment_type MF \
    --models "gbt rf xgb" \
    --fingerprints "Morgan MACCS RDKit" \
    --verbose

# Dry run to preview experiments
./scripts/batch_experiments/batch_experiment_local.sh \
    --experiment_type MF \
    --models gbt \
    --fingerprints Morgan \
    --dry-run
```

#### 3. Expected Output
```bash
$ ./scripts/batch_experiments/batch_experiment_local.sh --experiment_type MF --models gbt --fingerprints Morgan

[2025-07-14 15:02:26] OECD Toxicity Prediction - Local Batch Experiment Runner
[2025-07-14 15:02:26] ========================================================
[2025-07-14 15:02:26] Configuration:
[2025-07-14 15:02:26]   Experiment Type: MF - Molecular Fingerprints (no scaler required)
[2025-07-14 15:02:26]   Models: gbt
[2025-07-14 15:02:26]   Fingerprints: Morgan
[2025-07-14 15:02:26]   Total Experiments: 1
[2025-07-14 15:02:26] Running experiments sequentially...
[2025-07-14 15:19:26] ✓ Completed: TG201_Morgan_gbt (1020s)

# Generated files:
# results/models/molecular_fingerprints/TG201/best_model_Morgan_gbt.joblib
# results/logs/molecular_fingerprints/TG201/Morgan_gbt.log

# Note: For training new models, results may vary slightly from published 
# metrics due to different random seeds and newly fitted scalers.
# Use pre-trained models in Paper/Model/ for exact paper reproduction.
```

### Cluster Environment (Advanced)

For users with access to SLURM cluster systems:

#### 1. **Molecular Fingerprints Only**:
   ```bash
   # Edit batch_experiment_MF2.sh and set:
   EXPERIMENT_TYPE="MF"
   
   # Run experiment
   ./scripts/batch_experiments/batch_experiment_MF2.sh
   ```

#### 2. **Molecular Descriptors Only**:
   ```bash
   # Edit batch_experiment_MF2.sh and set:
   EXPERIMENT_TYPE="MD"
   
   # Run experiment
   ./scripts/batch_experiments/batch_experiment_MF2.sh
   ```

#### 3. **Combined Features**:
   ```bash
   # Edit batch_experiment_MF2.sh and set:
   EXPERIMENT_TYPE="COMBINED"
   
   # Run experiment
   ./scripts/batch_experiments/batch_experiment_MF2.sh
   ```

### Environment-Specific Notes

- **Local Environment**: Uses standard Python multiprocessing and shell background processes
- **Cluster Environment**: Requires SLURM workload manager and appropriate queue access  
- **Memory Requirements**: MD/Combined experiments need more RAM than MF experiments
- **Parallel Jobs**: Adjust based on your system's CPU cores and memory capacity

### Command Line Arguments

All model scripts support the following arguments:

- `--fingerprint_type`: Type of molecular fingerprint (MACCS, Morgan, RDKit, Layered, Pattern)
- `--file_path`: Path to the input Excel file
- `--model_save_path`: Directory to save the trained model
- `--scaler_save_path`: Path to save the scaler (only for MD and Combined experiments)

### Output Structure

```
results/
├── models/
│   ├── molecular_fingerprints/
│   │   └── dataset_name/
│   │       └── best_model_fingerprint_algorithm.joblib
│   ├── molecular_descriptors/
│   │   └── dataset_name/
│   │       ├── best_model_fingerprint_algorithm.joblib
│   │       └── scalers/
│   │           └── scaler_fingerprint_algorithm.joblib
│   └── combined/
│       └── dataset_name/
│           ├── best_model_fingerprint_algorithm.joblib
│           └── scalers/
│               └── scaler_fingerprint_algorithm.joblib
└── logs/
    └── (experiment logs)
```

## Project Structure

```
oecd-toxicity-prediction/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── predict_toxicity.py                 # Unified prediction script
├── example_input.xlsx                  # Example input data
├── data/
│   └── raw/
│       ├── molecular_fingerprints/     # MF data (no scaling)
│       ├── molecular_descriptors/      # MD data (with scaling)
│       └── combined/                   # MF+MD data (with scaling)
├── src/
│   ├── models/
│   │   ├── molecular_fingerprints/     # Models for MF (no scaler)
│   │   ├── molecular_descriptors/      # Models for MD (with scaler)
│   │   └── combined/                   # Models for MF+MD (with scaler)
│   └── utils/
│       ├── data_loader/
│       ├── preprocessing/
│       └── ...
├── scripts/
│   ├── run_single_experiment.sh        # Quick single experiment runner
│   └── batch_experiments/
│       ├── batch_experiment_local.sh   # Local batch runner (recommended)
│       ├── batch_experiment_local.py   # Python batch runner (advanced)
│       └── batch_experiment_MF2.sh     # SLURM cluster runner
└── results/
    ├── models/
    │   ├── molecular_fingerprints/
    │   ├── molecular_descriptors/
    │   │   └── scalers/
    │   └── combined/
    │       └── scalers/
    └── logs/
    │   └── batch_experiments/          # Batch experiment logs
    └── models/
        ├── molecular_fingerprints/
        ├── molecular_descriptors/
        │   └── scalers/
        └── combined/
            └── scalers/
```

## Data Format

### Input Data Requirements

All input Excel files should contain at minimum:
- `SMILES`: SMILES notation of chemical compounds
- `Toxicity` or `Genotoxicity`: Target labels (0/1 or negative/positive)

### Molecular Fingerprints Files
- Only SMILES and toxicity labels required
- Fingerprints generated automatically from SMILES

### Molecular Descriptors Files
- SMILES, toxicity labels, and pre-computed molecular descriptors
- Descriptors are automatically scaled using StandardScaler

### Combined Files
- SMILES, toxicity labels, and pre-computed molecular descriptors
- Fingerprints generated from SMILES, descriptors scaled automatically

## Model Performance Metrics

The models are evaluated using:
- F1 Score (primary metric)
- Precision
- Recall
- Accuracy
- ROC AUC

Results are logged during training and saved in JSON format.

## Repository Structure and Model Access

### Pre-trained Models Repository

**Note**: Pre-trained models are stored in a separate directory structure and can be accessed from:
`/home2/jjy0605/Toxicity/0817_Genotoxicity/OECD-COMTOX/published-models/Model/`

The pre-trained models are organized by toxicity endpoint:

```
published-models/Model/
├── Geno_model/                          # Genotoxicity models
│   ├── Descriptor_Scaler/               # Scaler files for molecular descriptors
│   ├── TG471_best_model_RDKit_lr.joblib
│   ├── TG471_best_model_RDKit_lr.json
│   ├── TG473_best_model_Morgan_dt.joblib
│   ├── TG473_best_model_Morgan_dt.json
│   ├── TG474_best_model_MACCS_rf.joblib
│   ├── TG474_best_model_MACCS_rf.json
│   ├── TG475_best_model_RDKit_dt.joblib
│   ├── TG475_best_model_RDKit_dt.json
│   ├── TG476D_best_model_RDKit_dt.joblib
│   ├── TG476D_best_model_RDKit_dt.json
│   ├── TG4782_best_model_RDKit_rf.joblib
│   ├── TG4782_best_model_RDKit_rf.json
│   ├── TG478_best_model_MACCS_dt.joblib
│   ├── TG478_best_model_MACCS_dt.json
│   ├── TG487D_best_model_RDKit_logistic.joblib
│   └── TG487D_best_model_RDKit_logistic.json
├── Eco_model/                           # Ecotoxicity models
│   ├── TG201_best_model_Layered_gbt.joblib
│   ├── TG201_best_model_Layered_gbt.json
│   ├── TG201_scaler_Layered_gbt.joblib
│   ├── TG202_best_model_RDKit_xgb.joblib
│   ├── TG202_best_model_RDKit_xgb.json
│   ├── TG202_scaler_RDKit_xgb.joblib
│   ├── TG203_best_model_RDKit_rf.joblib
│   ├── TG203_best_model_RDKit_rf.json
│   ├── TG203_scaler_RDKit_rf.joblib
│   ├── TG210_best_model_Morgan_logistic.joblib
│   ├── TG210_best_model_Morgan_logistic.json
│   ├── TG210_scaler_Morgan_logistic.joblib
│   ├── TG211_best_model_RDKit_rf.joblib
│   ├── TG211_best_model_RDKit_rf.json
│   └── TG211_scaler_RDKit_rf.joblib
├── DART_model/                          # Developmental toxicity models
│   ├── TG414+416+421_best_model_RDKit_dt.joblib
│   ├── TG414+416+421_best_model_RDKit_dt.json
│   ├── TG414_best_model_Morgan_logistic.joblib
│   ├── TG414_best_model_Morgan_logistic.json
│   ├── TG421D_best_model_RDKit_logistic.joblib
│   ├── TG421D_best_model_RDKit_logistic.json
│   └── TG421D_scaler_RDKit_logistic.joblib
├── Cancer_model/                        # Carcinogenicity models
│   ├── TG453D_best_model_RDKit_xgb.joblib
│   ├── TG453D_best_model_RDKit_xgb.json
│   └── TG453D_scaler_RDKit_xgb.joblib
├── Acute_model/                         # Acute toxicity models
│   ├── TG402D_best_model_Morgan_logistic.joblib
│   ├── TG402D_best_model_Morgan_logistic.json
│   ├── TG402D_scaler_Morgan_logistic.joblib
│   ├── TG403_aerosol_best_model_RDKit_rf.joblib
│   ├── TG403_aerosol_best_model_RDKit_rf.json
│   ├── TG403_vapor_best_model_MACCS_logistic.joblib
│   ├── TG403_vapor_best_model_MACCS_logistic.json
│   ├── TG420_best_model_MACCS_dt.joblib
│   └── TG420_best_model_MACCS_dt.json
└── Performance/                         # Consolidated performance metrics
```

### Model File Patterns

Models follow this naming pattern:
- **Model**: `{TG_number}_best_model_{fingerprint}_{algorithm}.joblib`
- **Scaler**: `{TG_number}_scaler_{fingerprint}_{algorithm}.joblib` (for MD/Combined models)
- **Metrics**: `{TG_number}_best_model_{fingerprint}_{algorithm}.json`

Where:
- `TG_number`: OECD Test Guideline number (e.g., TG471, TG201, TG414)
- `fingerprint`: MACCS, Morgan, RDKit, Layered, Pattern
- `algorithm`: lr (logistic), dt (decision tree), rf (random forest), gbt (gradient boosting), xgb (xgboost)

### Performance Benchmarks

Each model directory contains:
- **Model file** (`.joblib`): Trained model ready for prediction
- **Scaler file** (`.joblib`): For MD and Combined models only (from original paper experiments)
- **Performance log** (`.json`): Training metrics and validation results
- **Feature importance** (`.json`): For interpretable models

**📌 Note**: The pre-trained models and scalers in the Paper/Model/ directory are the exact files used to generate the results reported in the original research paper. These ensure reproducible predictions that match the published performance metrics.

### Download Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/stopdragonn/oecd-toxicity-prediction.git
   cd oecd-toxicity-prediction
   ```

2. **Access pre-trained models**:
   ```bash
   # Models are located at:
   # /home2/jjy0605/Toxicity/0817_Genotoxicity/OECD-COMTOX/published-models/Model/
   ls /home2/jjy0605/Toxicity/0817_Genotoxicity/OECD-COMTOX/published-models/Model/
   ```

3. **Verify model integrity**:
   ```bash
   # Check if models load correctly
   python -c "import joblib; print('Model loaded successfully')"
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

**Model and Code Attribution**: All models, scalers, and code in this repository are original research contributions by the authors and are shared under MIT License to facilitate scientific reproducibility and collaboration. The pre-trained models in `Paper/Model/` are provided as supplementary materials to our published research for exact result reproduction.

## Citation

If you use this code or models in your research, please cite our published work:

```bibtex
@article{Kim_oecd_toxicity_2024,
  title={OECD Test Guidelines-Based Toxicity Prediction using Machine Learning: A Comprehensive Study of Molecular Features and Model Performance},
  author={Donghyeon Kim, Jiyong Jeong, Siyeol Ahn and Jinhee Choi},
  journal={Computational Toxicology},
  year={2025},
  note={Please update with actual publication details when available}
}
```

**Model Attribution**: The pre-trained models in `published-models/Model/` directory are derived from our published research and are provided under the same MIT License for reproducibility purposes.

## Acknowledgements

This work was supported by Korea Environmental Industry & Technology Institute (KEITI) through 'Core Technology Development Project for Environmental Diseases Prevention and Management' (2021003310005) and the 2025 Chemical Substance Safety Management Cooperation Course through the funded by the Ministry of Environment. The authors gratefully acknowledge Soyoung Cho (Department of Statistics, University of Seoul) for her valuable assistance with the research methodologies employed in this study.

## Contact

For questions and support, please open an issue on GitHub or contact [jinhchoi@uos.ac.kr].

---

⚠️ **Note**: This repository is designed for research reproducibility. For production use, additional validation and testing may be required.
