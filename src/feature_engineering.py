"""
Feature Engineering Module for Credit Card Fraud Detection
==========================================================
Creates derived features to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for credit card transactions.
    
    Creates meaningful derived features from raw transaction data:
    - Time-based features (hour of day, is_night)
    - Amount-based features (amount categories, log transform)
    - Statistical features (z-scores, deviation from mean)
    - Interaction features
    """
    
    def __init__(self):
        self.amount_stats = {}  # Store training statistics
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'FeatureEngineer':
        """Fit feature engineer on training data to learn statistics."""
        self.amount_stats = {
            'mean': X['Amount'].mean() if 'Amount' in X.columns else 0,
            'std': X['Amount'].std() if 'Amount' in X.columns else 1,
            'median': X['Amount'].median() if 'Amount' in X.columns else 0,
            'q25': X['Amount'].quantile(0.25) if 'Amount' in X.columns else 0,
            'q75': X['Amount'].quantile(0.75) if 'Amount' in X.columns else 0,
        }
        self._is_fitted = True
        logger.info("FeatureEngineer fitted on training data")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Must call fit() first on training data.
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer not fitted. Call fit() on training data first.")
        
        X = X.copy()
        
        # Time-based features
        if 'Time' in X.columns:
            X = self._add_time_features(X)
        
        # Amount-based features
        if 'Amount' in X.columns:
            X = self._add_amount_features(X)
        
        # PCA feature interactions
        X = self._add_pca_interactions(X)
        
        logger.info(f"Feature engineering completed. New shape: {X.shape}")
        return X
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on data and transform in one step. Use only on training data."""
        return self.fit(X).transform(X)
    
    def _add_time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from the Time column."""
        # Time is in seconds from first transaction in the dataset
        # Convert to hour of day (assuming dataset starts at some point)
        X['Hour'] = (X['Time'] / 3600).astype(int) % 24
        
        # Is nighttime transaction (potentially more suspicious)
        X['Is_Night'] = ((X['Hour'] >= 22) | (X['Hour'] <= 5)).astype(int)
        
        # Is weekend-like pattern (simplified: transactions in off-peak hours)
        X['Is_OffPeak'] = ((X['Hour'] >= 0) & (X['Hour'] <= 7)).astype(int)
        
        # Time since start in hours (normalized)
        X['Time_Hours'] = X['Time'] / 3600
        
        # Cyclical encoding of hour
        X['Hour_Sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
        X['Hour_Cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
        
        return X
    
    def _add_amount_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
        # Log-transformed amount (handles skewness)
        X['Amount_Log'] = np.log1p(X['Amount'].clip(lower=0))
        
        # Amount categories
        X['Amount_Category'] = pd.cut(
            X['Amount'], 
            bins=[-np.inf, 5, 25, 100, 500, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
        
        # Z-score of amount (using training statistics)
        std = self.amount_stats['std'] if self.amount_stats['std'] > 0 else 1
        X['Amount_Zscore'] = (X['Amount'] - self.amount_stats['mean']) / std
        
        # Is the amount unusually high?
        X['Amount_IsHigh'] = (X['Amount'] > self.amount_stats['q75'] + 
                              1.5 * (self.amount_stats['q75'] - self.amount_stats['q25'])).astype(int)
        
        # Amount relative to median
        median = self.amount_stats['median'] if self.amount_stats['median'] > 0 else 1
        X['Amount_RelMedian'] = X['Amount'] / median
        
        return X
    
    def _add_pca_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features from the most important PCA components."""
        # Key discriminating PCA features based on typical fraud patterns
        important_features = ['V1', 'V3', 'V4', 'V7', 'V10', 'V12', 'V14', 'V17']
        available = [f for f in important_features if f in X.columns]
        
        if len(available) >= 2:
            # Magnitude of key fraud-indicative features
            X['V_FraudSignal_Mean'] = X[available].mean(axis=1)
            X['V_FraudSignal_Std'] = X[available].std(axis=1)
            
            # Specific interactions known to be useful
            if 'V1' in X.columns and 'V3' in X.columns:
                X['V1_V3_Interaction'] = X['V1'] * X['V3']
            
            if 'V14' in X.columns and 'V17' in X.columns:
                X['V14_V17_Interaction'] = X['V14'] * X['V17']
            
            if 'V10' in X.columns and 'V12' in X.columns:
                X['V10_V12_Interaction'] = X['V10'] * X['V12']
        
        return X
    
    def get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Return the list of feature names after transformation."""
        return list(self.transform(X).columns)


if __name__ == '__main__':
    # Quick test with synthetic data
    from data_preprocessing import generate_synthetic_data, DataPreprocessor
    
    # Generate and preprocess
    data_path = '../data/creditcard.csv'
    preprocessor = DataPreprocessor()
    
    import os
    if not os.path.exists(data_path):
        generate_synthetic_data(output_path=data_path)
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.run_pipeline(data_path)
    
    # Feature engineering
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_val_fe = fe.transform(X_val)
    X_test_fe = fe.transform(X_test)
    
    print(f"\nOriginal features: {X_train.shape[1]}")
    print(f"After feature engineering: {X_train_fe.shape[1]}")
    print(f"New features added: {X_train_fe.shape[1] - X_train.shape[1]}")
    print(f"\nNew feature names:")
    new_cols = [c for c in X_train_fe.columns if c not in X_train.columns]
    for col in new_cols:
        print(f"  - {col}")
