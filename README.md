# 🛡️ Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready machine learning system that detects fraudulent credit card transactions using ensemble models, advanced imbalance-handling techniques, and an interactive Streamlit dashboard with real-time explainability.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Dataset Specifications](#dataset-specifications)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Feature Engineering](#feature-engineering)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Model Development & Comparison](#model-development--comparison)
- [Model Evaluation & Business Impact](#model-evaluation--business-impact)
- [Explainability & Interpretability](#explainability--interpretability)
- [Web Dashboard (Streamlit)](#web-dashboard-streamlit)
- [Installation & Quickstart](#installation--quickstart)
- [Future Enhancements & Production Roadmap](#future-enhancements--production-roadmap)

---

## 🎯 Overview

Credit card fraud represents billions in global losses annually. This production-ready system tackles the core challenges of transaction risk modeling:
- **Extreme Class Imbalance:** Fraud represents ~0.172% of transactions (only 492 fraud cases in 284,807 transactions).
- **Asymmetric Cost of Errors:** Missed fraud (False Negatives) causes direct financial write-offs and reputational damage, whereas false alarms (False Positives) introduce friction for genuine cardholders.
- **Strict Leakage Prevention:** All scalers, distributions, and resampling techniques (SMOTE, undersampling) are fit strictly on training splits.
- **Operational Explainability:** Real-time transaction scoring accompanies feature importance and directional attribution.

### Target Users
- **Banks & Fintech Risk Teams:** Real-time transaction screening and rules engines.
- **Fraud Analysts:** Explainable alerts with risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and top driving factors.
- **ML Engineers:** Benchmark pipelines comparing Class Weighting, SMOTE, and Undersampling across tree-based ensembles.

---

## ✨ Key Features

1. **Modular Production Codebase:**
   - Dedicated modules for preprocessing, feature engineering, model training, evaluation, and inference.
2. **Imbalance-Aware Training:**
   - Evaluates Cost-Sensitive Class Weighting, SMOTE oversampling, and Random Undersampling.
3. **Multi-Model Suite:**
   - Benchmarks Logistic Regression (baseline), Random Forest, XGBoost, and LightGBM.
4. **Comprehensive Evaluation Suite:**
   - Confusion matrices, ROC curves, Precision-Recall curves (PR-AUC), and threshold sensitivity analysis.
5. **Interactive Streamlit Web Dashboard:**
   - Real-time transaction simulator, batch CSV scanner, interactive charts, and model comparison viewer.
6. **Interpretability Engine:**
   - Dynamic factor-attribution ranking features by importance and direction of risk.

---

## 🏗️ Project Architecture

```
credit-card-fraud-detection/
│
├── data/
│   ├── creditcard.csv                       # Kaggle ULB dataset (or generated synthetic replica)
│   └── prediction_history.json              # Persistent audit ledger of scored transactions
│
├── notebooks/
│   └── fraud_detection_eda_and_modeling.ipynb  # End-to-end EDA, experiments, and model diagnostics
│
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── data_preprocessing.py                # Cleaning, stratified splitting, RobustScaling (no leakage)
│   ├── feature_engineering.py               # Time, amount, and PCA interaction features
│   ├── train_model.py                       # RandomizedSearchCV, cross-validation, 3 imbalance strategies
│   ├── evaluate_model.py                    # Metric calculation, diagnostic plots, business impact report
│   ├── predict.py                           # Single/batch inference pipeline with explainability
│   └── history_store.py                     # Persistent audit ledger storage & retrieval
│
├── app/
│   └── app.py                               # AegisFraud Premium Fintech Streamlit Dashboard
│
├── models/
│   ├── best_model.joblib                    # Champion model artifact
│   ├── preprocessor.joblib                  # Fitted preprocessing pipeline
│   ├── feature_engineer.joblib              # Fitted feature engineering parameters
│   ├── model_comparison.csv                 # Detailed benchmark metrics table
│   └── training_metadata.joblib             # Model metadata, versioning, and hyperparameters
│
├── reports/
│   ├── confusion_matrix.png                 # Normalized & absolute confusion matrices
│   ├── roc_curve.png                        # ROC curve with AUC score
│   ├── precision_recall_curve.png           # PR curve with Average Precision (PR-AUC)
│   ├── feature_importance.png               # Top 20 predictive features
│   ├── threshold_analysis.png               # Precision, Recall & F1 vs. Decision Threshold
│   └── business_report.txt                  # Financial & operational impact summary
│
├── requirements.txt                         # Pinned dependencies
└── README.md                                # Comprehensive documentation
```

---

## 📊 Dataset Specifications

### Kaggle ULB Credit Card Fraud Dataset
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (`0.172%`)
- **Legitimate Transactions:** 284,315 (`99.828%`)
- **Features:**
  - `Time`: Elapsed seconds since the first transaction in the dataset.
  - `V1 - V28`: Principal Component Analysis (PCA) transformed features preserving cardholder confidentiality.
  - `Amount`: Transaction transaction amount in USD.
  - `Class`: Ground-truth label (`1 = Fraud`, `0 = Legitimate`).

*(Note: The codebase includes an automated synthetic generator `generate_synthetic_data()` reproducing the exact schema, distribution shifts, and correlations if testing offline without immediate Kaggle downloads.)*

---

## ⚙️ Data Preprocessing Pipeline

To guarantee statistical validity and prevent data leakage:
1. **Deduplication & Quality Checks:** Detects and handles duplicate records and validates data types.
2. **Stratified Splitting:** 
   - 70% Training / 10% Validation / 20% Test.
   - Preserves minority class percentage uniformly across splits (`stratify=y`).
3. **Outlier-Resistant Scaling:**
   - `RobustScaler` scales `Time` and `Amount` using median and interquartile range (IQR), preventing extreme fraud amounts from skewing scaling parameters.
   - PCA components `V1-V28` remain untouched since they are already zero-centered and normalized.
   - Scaler is fitted exclusively on `X_train` and applied down-pipeline to `X_val` and `X_test`.

---

## 🔧 Feature Engineering

The `FeatureEngineer` class creates domain-specific signals without introducing lookahead bias:

| Engineered Feature | Domain Category | Analytical Purpose |
|-------------------|-----------------|--------------------|
| `Hour` | Temporal | Extracted transaction hour (0–23) |
| `Is_Night` | Temporal | Flags transactions between 10 PM and 5 AM (higher risk window) |
| `Is_OffPeak` | Temporal | Flags transactions between 12 AM and 7 AM |
| `Hour_Sin`, `Hour_Cos` | Temporal | Cyclical trigonometric encoding of the 24-hour cycle |
| `Amount_Log` | Monetary | Log-transform (`log1p`) mitigating positive skewness of amounts |
| `Amount_Category` | Monetary | Discretized spending buckets (micro, small, medium, large, extreme) |
| `Amount_Zscore` | Monetary | Deviation from training mean normalized by standard deviation |
| `Amount_IsHigh` | Monetary | Binary flag for amounts exceeding the 75th percentile + 1.5 * IQR |
| `Amount_RelMedian` | Monetary | Ratio of transaction amount to the median training amount |
| `V_FraudSignal_Mean` | Behavioral PCA | Aggregate mean of key discriminating components (`V1, V3, V4, V7, V10, V12, V14, V17`) |
| `V_FraudSignal_Std` | Behavioral PCA | Dispersion across key discriminating components |
| `V1_V3_Interaction` | Cross-Feature | Non-linear interaction between negative deflection signals |
| `V14_V17_Interaction` | Cross-Feature | Interaction between top fraud-correlated PCA features |
| `V10_V12_Interaction` | Cross-Feature | Interaction measuring co-occurrence anomalies |

---

## ⚖️ Handling Class Imbalance

We systematically compare three distinct imbalance strategies:

1. **Cost-Sensitive Class Weighting:**
   - Inverts class frequencies to heavily penalize fraud misclassification directly within the loss function (`class_weight='balanced'` or `scale_pos_weight`).
2. **Synthetic Minority Over-sampling (SMOTE):**
   - Synthesizes synthetic minority instances by interpolating between nearest neighbors in feature space. Applied **strictly to training sets**.
3. **Random Undersampling:**
   - Selectively downsizes the majority class to achieve a controlled target ratio (e.g. 2:1 non-fraud to fraud), minimizing training overhead while preserving fraud cases.

---

## 🤖 Model Development & Comparison

Models trained across all strategies using `StratifiedKFold` cross-validation:
- **Logistic Regression (Baseline):** Linear decision boundary with $L_1$/$L_2$ regularization.
- **Random Forest:** Ensemble of decorrelated decision trees with bootstrap aggregation.
- **XGBoost:** Regularized gradient boosting optimizing binary log-loss.
- **LightGBM:** Fast, leaf-wise gradient boosting optimized for large tabular datasets.

### Metrics Focus
Because overall accuracy on this dataset exceeds 99.8% even for a naive zero-classifier, we optimize specifically for:
- **Precision-Recall AUC (PR-AUC):** High area under the precision-recall curve reflects strong discrimination in severe imbalance.
- **Fraud Recall (Sensitivity):** Fraction of actual fraudulent transactions detected.
- **Precision:** Fraction of flagged transactions that are genuinely fraudulent.
- **F1-Score:** Harmonic balance between precision and recall.

---

## 📊 Model Evaluation & Business Impact

### Threshold Sensitivity Analysis
The standard 0.5 decision threshold is rarely optimal for financial risk. Our evaluation pipeline systematically calculates precision, recall, and F1 across thresholds from 0.10 to 0.90 to identify the economic optimum:
- **Lower Threshold (e.g. 0.25 - 0.35):** Maximizes recall, capturing more fraud at the expense of manual review volume.
- **Higher Threshold (e.g. 0.65 - 0.75):** Minimizes false positives, ideal when review capacity is constrained or automated transaction blocking would cause user friction.

### Business Trade-off Framework
- **Missed Fraud Loss:** $\text{FN} \times \text{Average Fraud Amount}$
- **False Alarm Operational Cost:** $\text{FP} \times (\text{Customer Verification Cost} + \text{Churn Probability})$

---

## 🔍 Explainability & Interpretability

1. **Global Feature Importance:**
   - Tree-based Gini gain / split frequency importance identifies top diagnostic features across the portfolio (typically `V14`, `V10`, `V12`, `V17`, `Amount_Log`).
2. **Local Prediction Attribution:**
   - For every flagged transaction, the prediction service provides:
     - Exact fraud probability percentage.
     - Assigned risk tier (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
     - Top 5 driving factors with individual magnitude and directional impact ("increases fraud risk" vs. "decreases fraud risk").

---

## 🖥️ Web Dashboard (Streamlit)

The web application (`app/app.py`) provides an interactive interface:
- **🏠 Dashboard:** Portfolio overview showing transaction totals, fraud percentage, distribution histograms, time-of-day fraud activity, and correlation matrices.
- **🔍 Predict Transaction:**
  - *Sample Transaction Loader:* Test pre-configured normal vs. suspicious transaction patterns with one click.
  - *Manual Feature Entry:* Adjust individual amounts, timestamps, and PCA sliders to test edge cases.
  - *Batch CSV Scanner:* Upload batches of transactions to receive immediate risk classification and downloadable scoring logs.
- **📊 Model Performance:** Interactive model comparison tables, precision-recall scatter plots, radar capability charts, and embedded diagnostic plots.
- **📖 About:** Architecture breakdown, documentation references, and technical specifications.

---

## 🚀 Installation & Quickstart

### 1. Prerequisites
- Python 3.9 - 3.12
- Git

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Pipeline End-to-End
```bash
# Step 1: Preprocess data (or generate synthetic dataset)
python src/data_preprocessing.py

# Step 2: Train all models across imbalance strategies
python src/train_model.py

# Step 3: Run comprehensive evaluation and generate report visualizations
python src/evaluate_model.py

# Step 4: Test single transaction inference
python src/predict.py
```

### 4. Launch Streamlit Dashboard
```bash
# Windows PowerShell:
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
streamlit run app/app.py

# Linux / macOS:
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔮 Future Enhancements & Production Roadmap

- [ ] **Streaming Ingestion:** Real-time event streaming pipeline using Apache Kafka or AWS Kinesis.
- [ ] **Graph Neural Networks (GNN):** Account-device-merchant graph embeddings to detect organized fraud rings.
- [ ] **Concept Drift & Data Drift Detection:** Automated PSI (Population Stability Index) and Kolmogorov-Smirnov drift monitors.
- [ ] **Containerization & CI/CD:** Production Dockerfile and GitHub Actions pipeline for automated unit tests and model artifact versioning (MLflow/DVC).
- [ ] **Active Learning Loop:** Automated retraining queues routing borderline transactions (probabilities between 0.40 and 0.60) to fraud analysts for manual labeling.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
