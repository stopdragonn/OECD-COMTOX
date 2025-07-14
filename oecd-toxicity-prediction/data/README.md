# Data Description

This directory contains the datasets used for OECD toxicity prediction experiments across multiple endpoints including genotoxicity, carcinogenicity, acute toxicity, developmental and reproductive toxicity (DART), and ecotoxicity.

## Data Organization

### Raw Data Structure

```
data/raw/
├── molecular_fingerprints/     # MF experiments (no scaling)
├── molecular_descriptors/      # MD experiments (with scaling)
└── combined/                   # MF+MD experiments (with scaling)
```

## Dataset Types

### 1. Molecular Fingerprints (MF)
- **Directory**: `molecular_fingerprints/`
- **File format**: `TGXXX.xlsx`
- **Required columns**:
  - `SMILES`: SMILES notation of chemical compounds
  - `Toxicity` or `Genotoxicity`: Binary labels (0/1 or negative/positive)
- **Preprocessing**: None (fingerprints generated from SMILES)
- **Scaling**: Not required

### 2. Molecular Descriptors (MD)
- **Directory**: `molecular_descriptors/`
- **File format**: `TGXXX_Descriptor_desalt.xlsx`
- **Required columns**:
  - `SMILES`: SMILES notation
  - `Toxicity`, `Genotoxicity`, or endpoint-specific labels: Binary labels
  - Additional columns: Pre-computed molecular descriptors
- **Preprocessing**: StandardScaler applied to descriptor columns
- **Scaling**: Required

### 3. Combined Features (MF+MD)
- **Directory**: `combined/`
- **File format**: `TGXXX_Descriptor_desalt_YYMMDD.xlsx`
- **Required columns**:
  - `SMILES`: SMILES notation
  - `Toxicity`, `Genotoxicity`, or endpoint-specific labels: Binary labels
  - Additional columns: Pre-computed molecular descriptors
- **Preprocessing**: 
  - Fingerprints generated from SMILES
  - StandardScaler applied to descriptor columns only
- **Scaling**: Required (descriptors only)

## Data Sources

The datasets are derived from OECD test guidelines (TG) covering multiple toxicity endpoints:

### Genotoxicity
- **TG 471**: Bacterial reverse mutation test (Ames test)
- **TG 473**: In vitro mammalian chromosomal aberration test
- **TG 474**: In vivo mammalian erythrocyte micronucleus test
- **TG 476**: In vitro mammalian cell gene mutation tests
- **TG 478**: Rodent dominant lethal test
- **TG 486**: Unscheduled DNA synthesis test
- **TG 487**: In vitro mammalian cell micronucleus test

### Carcinogenicity
- **TG 451**: Carcinogenicity studies
- **TG 453**: Combined chronic toxicity/carcinogenicity studies

### Acute Toxicity
- **TG 420**: Acute oral toxicity - fixed dose procedure
- **TG 423**: Acute oral toxicity - acute toxic class method
- **TG 425**: Acute oral toxicity - up-and-down procedure

### Developmental and Reproductive Toxicity (DART)
- **TG 414**: Prenatal developmental toxicity study
- **TG 416**: Two-generation reproduction toxicity study
- **TG 421**: Reproduction/developmental toxicity screening test
- **TG 422**: Combined repeated dose toxicity study with reproduction/developmental toxicity screening test
- **TG 443**: Extended one-generation reproductive toxicity study

### Ecotoxicity
- **TG 201**: Freshwater alga and cyanobacteria, growth inhibition test
- **TG 202**: Daphnia sp. acute immobilisation test
- **TG 203**: Fish, acute toxicity test
- **TG 210**: Fish, early-life stage toxicity test
- **TG 211**: Daphnia magna reproduction test

## File Naming Convention

- `TGXXX.xlsx`: Molecular fingerprint datasets
- `TGXXX_Descriptor_desalt.xlsx`: Molecular descriptor datasets
- `TGXXX_Descriptor_desalt_YYMMDD.xlsx`: Combined datasets with date stamps

## Data Processing Notes

### Label Handling
Labels are automatically converted:
- String format: `"negative"` → 0, `"positive"` → 1
- Numeric format: Used as-is (0/1)

### Missing Data
- Invalid SMILES are automatically detected and removed
- Corresponding labels are also removed to maintain data integrity

### Column Exclusions
The following columns are automatically excluded from feature extraction:
- `SMILES`: Used for fingerprint generation, not as features
- `Toxicity`/`Genotoxicity`/endpoint-specific labels: Target labels
- `DTXSID`: Chemical identifiers
- `Chemical`: Chemical names
- `CasRN`: CAS registry numbers

## Usage Examples

### Loading MF Data
```python
from src.utils.data_loader.mf_loader import load_data

# Load molecular fingerprints data (no scaling)
X, y = load_data("data/raw/molecular_fingerprints/TG201.xlsx", 
                 fingerprint_type="Morgan")
```

### Loading MD Data
```python
from src.utils.data_loader.md_loader import load_data

# Load molecular descriptors data (with scaling)
X, y, scaler = load_data("data/raw/molecular_descriptors/TG201_Descriptor_desalt.xlsx",
                         scaler_save_path="results/scalers/scaler.joblib")
```

### Loading Combined Data
```python
from src.utils.data_loader.combined_loader import load_data

# Load combined features data (with scaling for descriptors)
X, y, scaler = load_data("data/raw/combined/TG414_Descriptor_desalt_250501.xlsx",
                         fingerprint_type="Morgan",
                         scaler_save_path="results/scalers/scaler.joblib")
```

## Data Availability

For data availability and access:
1. Small sample datasets are included in this repository
2. Full datasets may be available upon request
3. Large datasets may be hosted on external repositories (Zenodo, Figshare, etc.)

## Privacy and Ethics

All data used in this research:
- Contains no personally identifiable information
- Uses publicly available chemical structures and properties
- Follows appropriate data sharing guidelines
- Complies with institutional data management policies
