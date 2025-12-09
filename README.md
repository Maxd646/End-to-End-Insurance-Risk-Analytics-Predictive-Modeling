# Insurance Risk Analytics & Predictive Modeling — ACIS Project

A complete, modular, production-ready machine learning pipeline for
insurance analytics.

---

## Project Overview

This project implements a **fully modular end-to-end ML pipeline** for
insurance risk analytics and predictive modeling. It supports real-world
insurance business applications such as:

- Analyzing historical policies, claims, and exposures\
- Performing EDA and anomaly detection\
- Conducting hypothesis tests to validate key risk drivers\
- Building models for claim probability, claim severity, and premium
  optimization\
- End-to-end reproducible ML pipeline with CI/CD support\
- Integrated reporting, logging, and versioning

---

## Business Objective

**AlphaCare Insurance Solutions (ACIS)** aims to:

- Identify **low-risk customer segments**\
- Optimize **premium pricing** while maximizing profitability\
- Understand **factors contributing to claims**\
- Support **actuarial and underwriting decisions**\
- Enhance customer retention with targeted strategies

---

## Full Project Folder Structure

    End-to-End-Insurance-Risk-Analytics-Predictive-Modeling/
    ├── .github/
    │   └── workflows/                 # CI/CD pipelines (tests, linting, dvc)
    ├── configs/
    │   ├── data.yaml                  # Dataset configuration
    │   ├── dvc_remote.yaml            # DVC remote configuration
    │   ├── logs.yaml                  # Logging settings
    │   └── modeling.yaml              # ML model configurations
    ├── data/
    │   ├── raw/                       # Original data
    │   ├── processed/                 # Cleaned & feature engineered data
    ├── docs/                          # Documentation & reports
    ├── notebooks/
    │   ├── analysis/
    │   │   ├── hypothesis_tests.ipynb
    │   │   └── model_building.ipynb
    │   └── exploration/
    │       ├── data_overview.ipynb
    │       └── eda.ipynb
    ├── scripts/
    │   ├── __init__.py
    │   ├── clean_data.py
    │   ├── run_eda_pipeline.py
    │   ├── run_hypothesis_tests.py
    │   └── train_models.py
    ├── src/
    │   └── insurance_analytics/
    │       ├── __init__.py
    │       ├── core/
    │       │   ├── __init__.py
    │       │   ├── config.py
    │       │   ├── logger.py
    │       │   ├── registry.py
    │       │   └── scheduler.py
    │       ├── eda/
    │       │   ├── __init__.py
    │       │   ├── exploration.py
    │       │   └── visualization.py
    │       ├── models/
    │       │   ├── __init__.py
    │       │   ├── evaluation.py
    │       │   ├── interpretability.py
    │       │   ├── linear_regression.py
    │       │   ├── random_forest.py
    │       │   └── xgboost_model.py
    │       ├── preprocessing/
    │       │   ├── __init__.py
    │       │   ├── cleaner.py
    │       │   └── feature_engineering.py
    │       ├── utils/
    │       │   ├── __init__.py
    │       │   ├── io_utils.py
    │       │   ├── metrics.py
    │       │   ├── project_root.py
    │       │   ├── system.py
    │       │   └── validation.py
    │       └── viz/
    │           ├── __init__.py
    │           └── plots.py
    ├── tests/
    │   ├── integration/
    │   │   ├── __init__.py
    │   │   ├── test_dvc_integration.py
    │   │   ├── test_eda_pipeline.py
    │   │   ├── test_full_pipeline.py
    │   │   └── test_model_pipeline.py
    │   └── unit/
    │       ├── __init__.py
    │       ├── test_cleaners.py
    │       ├── test_features.py
    │       ├── test_hypothesis.py
    │       ├── test_loaders.py
    │       ├── test_models.py
    │       └── test_registry.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt

---

## How to Run the Project

### Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Data Cleaning

```bash
python scripts/clean_data.py
```

### Run EDA Pipeline

```bash
python scripts/run_eda_pipeline.py
```

### Train Machine Learning Models

```bash
python scripts/train_models.py
```

### Use Jupyter Notebooks

```bash
jupyter notebook
```

---

## Key Features

- ✔ Modular ML architecture\
- ✔ Clear data/configs/scripts separation\
- ✔ DVC versioning\
- ✔ CI-ready workflows\
- ✔ Logging & validation utilities\
- ✔ Interpretability (SHAP, feature importance)\
- ✔ Reproducible experiments

---

## Reports

- `docs/interim_report.md`\
- `docs/final_report.md`

---

## Testing

```bash
pytest
```

---

## Version Control

```bash
git init
git add .
git commit -m "Initial project structure"
```
