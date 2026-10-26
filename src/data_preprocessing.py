"""
Data Preprocessing Module for Credit Card Fraud Detection
=========================================================
Handles loading, cleaning, splitting, and scaling of the credit card transaction dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Dict, Optional
import logging
import os
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    End-to-end data preprocessing pipeline for credit card fraud detection.
    
    Handles:
    - Data loading and inspection
    - Missing value treatment
    - Duplicate removal
    - Outlier detection and handling
    - Feature scaling (fit only on training data to avoid leakage)
    - Train/validation/test splitting
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_columns = None
        self.target_column = 'Class'
        self._is_fitted = False
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the credit card transaction dataset."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Fraud cases: {df[self.target_column].sum()} ({df[self.target_column].mean()*100:.4f}%)")
        return df
    
    def inspect_data(self, df: pd.DataFrame) -> Dict:
        """Generate a comprehensive data quality report."""
        report = {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'total_missing': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'class_distribution': df[self.target_column].value_counts().to_dict(),
            'fraud_rate': df[self.target_column].mean(),
            'basic_stats': df.describe()
        }
        logger.info(f"Data Quality Report:")
        logger.info(f"  Shape: {report['shape']}")
        logger.info(f"  Total missing values: {report['total_missing']}")
        logger.info(f"  Duplicates: {report['duplicates']}")
        logger.info(f"  Fraud rate: {report['fraud_rate']*100:.4f}%")
        return report
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset: handle missing values, remove duplicates."""
        df = df.copy()
        
        # Remove duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            df = df.drop_duplicates()
            logger.info(f"Removed {n_duplicates} duplicate rows")
        
        # Handle missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
                    logger.info(f"Filled {missing[col]} missing values in {col} with median")
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Split data into train/validation/test sets with stratification.
        
        IMPORTANT: Stratified split preserves the fraud ratio in each set.
        """
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {X_train.shape[0]} samples ({y_train.mean()*100:.4f}% fraud)")
        logger.info(f"  Validation: {X_val.shape[0]} samples ({y_val.mean()*100:.4f}% fraud)")
        logger.info(f"  Test: {X_test.shape[0]} samples ({y_test.mean()*100:.4f}% fraud)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on training data and transform. Only call on training data!"""
        # Scale 'Amount' and 'Time' columns (V1-V28 are already PCA-scaled)
        cols_to_scale = [col for col in ['Time', 'Amount'] if col in X_train.columns]
        
        if cols_to_scale:
            X_train = X_train.copy()
            X_train[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
            self._is_fitted = True
            logger.info(f"Fitted and transformed columns: {cols_to_scale}")
        
        return X_train
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the already-fitted scaler. For val/test data."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted yet. Call fit_transform on training data first.")
        
        cols_to_scale = [col for col in ['Time', 'Amount'] if col in X.columns]
        
        if cols_to_scale:
            X = X.copy()
            X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])
        
        return X
    
    def save_preprocessor(self, filepath: str):
        """Save the fitted preprocessor (scaler + metadata) for deployment."""
        artifacts = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'is_fitted': self._is_fitted
        }
        joblib.dump(artifacts, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str) -> 'DataPreprocessor':
        """Load a previously saved preprocessor."""
        artifacts = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.scaler = artifacts['scaler']
        preprocessor.feature_columns = artifacts['feature_columns']
        preprocessor.target_column = artifacts['target_column']
        preprocessor._is_fitted = artifacts['is_fitted']
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
    
    def run_pipeline(self, filepath: str) -> Tuple:
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, report)
        """
        # Load and inspect
        df = self.load_data(filepath)
        report = self.inspect_data(df)
        
        # Clean
        df = self.clean_data(df)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # Scale (fit only on training data)
        X_train = self.fit_transform(X_train)
        X_val = self.transform(X_val)
        X_test = self.transform(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, report


def generate_synthetic_data(n_samples: int = 284807, fraud_ratio: float = 0.00173, 
                            output_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic credit card fraud data mimicking the Kaggle ULB dataset structure.
    
    This is useful for development/testing when the real dataset isn't available.
    The real dataset should be downloaded from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
    Args:
        n_samples: Total number of transactions
        fraud_ratio: Fraction of fraudulent transactions
        output_path: If provided, save the CSV to this path
        
    Returns:
        DataFrame with synthetic credit card transaction data
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    logger.info(f"Generating synthetic data: {n_legitimate} legitimate, {n_fraud} fraud transactions")
    
    # Generate Time feature (seconds elapsed, spanning ~2 days)
    time_legitimate = np.random.uniform(0, 172800, n_legitimate)
    time_fraud = np.random.uniform(0, 172800, n_fraud)
    
    # Generate V1-V28 PCA features
    # Legitimate transactions: centered around 0 with unit variance
    V_legitimate = np.random.randn(n_legitimate, 28)
    
    # Fraud transactions: shifted distributions for some features
    V_fraud = np.random.randn(n_fraud, 28)
    # Key discriminating features (based on real dataset patterns)
    V_fraud[:, 0] -= 2.5   # V1 tends to be more negative for fraud
    V_fraud[:, 1] += 1.5   # V2 shifted
    V_fraud[:, 2] -= 3.0   # V3 strongly negative for fraud
    V_fraud[:, 3] += 1.0   # V4 shifted
    V_fraud[:, 6] -= 2.0   # V7 shifted
    V_fraud[:, 9] -= 3.5   # V10 strongly negative for fraud
    V_fraud[:, 11] += 2.0  # V12 shifted
    V_fraud[:, 13] -= 2.5  # V14 strongly negative for fraud
    V_fraud[:, 15] -= 2.0  # V16 shifted
    V_fraud[:, 16] -= 3.0  # V17 strongly negative for fraud
    
    # Generate Amount feature
    amount_legitimate = np.abs(np.random.exponential(scale=88.0, size=n_legitimate))
    amount_fraud = np.abs(np.random.exponential(scale=122.0, size=n_fraud))
    
    # Create DataFrames
    v_columns = [f'V{i}' for i in range(1, 29)]
    
    legitimate_df = pd.DataFrame(V_legitimate, columns=v_columns)
    legitimate_df['Time'] = time_legitimate
    legitimate_df['Amount'] = amount_legitimate
    legitimate_df['Class'] = 0
    
    fraud_df = pd.DataFrame(V_fraud, columns=v_columns)
    fraud_df['Time'] = time_fraud
    fraud_df['Amount'] = amount_fraud
    fraud_df['Class'] = 1
    
    # Combine and shuffle
    df = pd.concat([legitimate_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reorder columns to match real dataset
    cols = ['Time'] + v_columns + ['Amount', 'Class']
    df = df[cols]
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic dataset saved to {output_path}")
    
    return df


if __name__ == '__main__':
    # Generate synthetic data for development
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'creditcard.csv')
    df = generate_synthetic_data(output_path=data_path)
    
    # Run preprocessing pipeline
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, report = preprocessor.run_pipeline(data_path)
    
    # Save preprocessor
    preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'preprocessor.joblib')
    preprocessor.save_preprocessor(preprocessor_path)
    
    print("\nPreprocessing pipeline completed successfully!")
    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Test features shape: {X_test.shape}")
