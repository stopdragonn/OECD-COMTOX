# OECD-COMTOX: Computational Toxicology Research Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2020.03+-green.svg)](https://www.rdkit.org/)
[![Research](https://img.shields.io/badge/Research-Computational%20Toxicology-blue.svg)](https://github.com/stopdragonn/OECD-COMTOX)

**A comprehensive computational toxicology research repository for OECD Test Guidelines-based toxicity prediction using machine learning.**

## 🔬 Overview

This repository contains the complete research framework for predicting chemical toxicity endpoints based on OECD Test Guidelines (TG) using advanced machine learning approaches. Our work focuses on developing reliable, interpretable, and reproducible models for various toxicity endpoints including genotoxicity, carcinogenicity, acute toxicity, developmental toxicity, and ecotoxicity.

### 🎯 Key Features

- **Multiple Toxicity Endpoints**: Comprehensive coverage of OECD TG endpoints
- **Diverse Molecular Representations**: Fingerprints, descriptors, and combined features
- **Reproducible Research**: Fully documented and tested codebase
- **Pre-trained Models**: Ready-to-use models with published performance
- **Interpretable AI**: SHAP-based model interpretability analysis
- **User-Friendly**: Both local and cluster execution support

## 📁 Repository Structure

```
OECD-COMTOX/
├── oecd-toxicity-prediction/     # Main codebase for model training and prediction
│   ├── src/                      # Source code
│   │   ├── models/              # ML model implementations
│   │   └── utils/               # Utility functions and data loaders
│   ├── scripts/                 # Batch experiment and execution scripts
│   ├── data/                    # Training and test datasets
│   └── results/                 # Model outputs and logs
├── published-models/             # Pre-trained models and research assets
│   ├── Model/                   # Published model files (.joblib)
│   │   ├── Geno_model/         # Genotoxicity models
│   │   ├── Eco_model/          # Ecotoxicity models
│   │   ├── DART_model/         # Developmental toxicity models
│   │   ├── Cancer_model/       # Carcinogenicity models
│   │   └── Acute_model/        # Acute toxicity models
│   ├── Data/                   # Original research datasets
│   └── SHAP/                   # Model interpretability analysis
└── LICENSE                     # MIT License
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/stopdragonn/OECD-COMTOX.git
cd OECD-COMTOX
```

### 2. Set Up Environment
```bash
cd oecd-toxicity-prediction
conda env create -f environment.yml
conda activate oecd-toxicity-pred
```

### 3. Run Your First Prediction
```bash
# Test a quick experiment
./scripts/run_single_experiment.sh --type MF --model gbt --fingerprint Morgan --data TG201
```

### 4. Use Pre-trained Models
```bash
# Use published models for immediate predictions
python predict_toxicity.py \
    --model_type MF \
    --fingerprint_type RDKit \
    --input_file your_compounds.xlsx \
    --model_path published-models/Model/Geno_model/TG471_best_model_RDKit_lr.joblib
```

## 📊 Supported Toxicity Endpoints

| Endpoint Category | OECD Test Guidelines | Models Available |
|-------------------|---------------------|------------------|
| **Genotoxicity** | TG 471, 473, 474, 475, 476, 478, 486, 487 | ✅ |
| **Carcinogenicity** | TG 451, 453 | ✅ |
| **Acute Toxicity** | TG 402, 403, 420, 423, 425 | ✅ |
| **Developmental Toxicity** | TG 414, 416, 421, 422, 443 | ✅ |
| **Ecotoxicity** | TG 201, 202, 203, 210, 211 | ✅ |

## 🧪 Molecular Feature Types

- **Molecular Fingerprints (MF)**: MACCS, Morgan, RDKit, Layered, Pattern
- **Molecular Descriptors (MD)**: RDKit, Mordred descriptors with scaling
- **Combined Features (MF+MD)**: Integrated fingerprint and descriptor approaches

## 🎯 Machine Learning Models

- Gradient Boosting (GBT)
- Random Forest (RF)
- XGBoost (XGB)
- Logistic Regression
- Decision Tree (DT)
- Multi-layer Perceptron (MLP)
- Linear/Quadratic Discriminant Analysis (LDA/QDA)
- Partial Least Squares Discriminant Analysis (PLS-DA)
- LightGBM (LGB)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Model Attribution

All models and code in this repository are original research contributions shared under MIT License for scientific reproducibility and collaboration. Pre-trained models in `published-models/` are provided as supplementary materials for exact result reproduction.

## 📖 Citation

If you use this repository in your research, please cite our work:

```bibtex
@article{Kim_oecd_toxicity_2024,
  title={OECD Test Guidelines-Based Toxicity Prediction using Machine Learning: A Comprehensive Study of Molecular Features and Model Performance},
  author={Donghyeon Kim, Jiyong Jeong, Siyeol Ahn and Jinhee Choi},
  journal={Computational Toxicology},
  year={2025},
  doi={10.1016/j.comtox.2025.100369},
  url={https://doi.org/10.1016/j.comtox.2025.100369}
}
```

## 🙏 Acknowledgements

This work was supported by Korea Environmental Industry & Technology Institute (KEITI) through 'Core Technology Development Project for Environmental Diseases Prevention and Management' (2021003310005) and by the Specialized Graduate Program for Chemical Substance Safety Management funded by the Ministry of Environment. The authors acknowledge the Urban Big Data and AI Institute of the University of Seoul supercomputing resources (http://ubai.uos.ac.kr) made available for conducting the research reported in this paper. The authors gratefully acknowledge Soyoung Cho (Department of Statistics, University of Seoul) for her valuable assistance with the research methodologies employed in this study.

## 📞 Contact

- **Primary Contact**: [jinhchoi@uos.ac.kr](mailto:jinhchoi@uos.ac.kr)
- **GitHub**: [@stopdragonn](https://github.com/stopdragonn)
- **Lab Website**: [Environmental Systems Toxicology Lab](https://est.uos.ac.kr/)

---

⚠️ **Note**: This repository is designed for research and educational purposes. For production use in regulatory contexts, additional validation may be required.

🔬 **Research Excellence**: Advancing computational toxicology through open science and reproducible research.
