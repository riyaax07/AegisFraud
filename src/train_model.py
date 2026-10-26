"""
Model Training Module for Credit Card Fraud Detection
=====================================================
Trains and compares multiple ML models with imbalance-handling techniques.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    roc_auc_score, average_precision_score, make_scorer
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Dict, Tuple, Optional, Any
import joblib
import os
import time
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """
    Trains multiple fraud detection models and compares their performance.
    
    Implements three imbalance-handling strategies:
    1. Class weighting (built into model)
    2. SMOTE oversampling
    3. Random undersampling
    
    Models trained:
    - Logistic Regression (baseline)
    - Random Forest
    - XGBoost
    - LightGBM
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
    
    def _get_base_models(self) -> Dict:
        """Define base model configurations."""
        return {
            'Logistic Regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state
                ),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'class_weight': ['balanced', None],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15],
                    'min_samples_split': [2, 5],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'min_child_weight': [1, 3, 5],
                    'scale_pos_weight': [1, 10, 50, 100],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 100],
                    'is_unbalance': [True, False],
                    'min_child_samples': [5, 10, 20],
                    'subsample': [0.7, 0.8, 0.9]
                }
            }
        }
    
    def train_with_class_weight(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series,
                                 n_iter: int = 20) -> Dict:
        """
        Strategy 1: Train models using built-in class weighting.
        """
        logger.info("=" * 60)
        logger.info("STRATEGY 1: Class Weighting")
        logger.info("=" * 60)
        
        results = {}
        models = self._get_base_models()
        
        for name, config in models.items():
            logger.info(f"\nTraining {name} with class weighting...")
            start_time = time.time()
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scorer = make_scorer(f1_score, zero_division=0)
            
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=min(n_iter, self._count_combinations(config['params'])),
                scoring=scorer,
                cv=cv,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            
            train_time = time.time() - start_time
            model = search.best_estimator_
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            result = {
                'model': model,
                'strategy': 'class_weight',
                'best_params': search.best_params_,
                'train_time': train_time,
                'val_precision': precision_score(y_val, y_pred, zero_division=0),
                'val_recall': recall_score(y_val, y_pred, zero_division=0),
                'val_f1': f1_score(y_val, y_pred, zero_division=0),
                'val_roc_auc': roc_auc_score(y_val, y_proba),
                'val_pr_auc': average_precision_score(y_val, y_proba),
                'cv_score': search.best_score_
            }
            
            results[f"{name} (Weighted)"] = result
            self._log_result(f"{name} (Weighted)", result)
            
            # Track best model
            if result['val_f1'] > self.best_score:
                self.best_score = result['val_f1']
                self.best_model = model
                self.best_model_name = f"{name} (Weighted)"
        
        self.results.update(results)
        return results
    
    def train_with_smote(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Strategy 2: Train models using SMOTE oversampling on training data only.
        """
        logger.info("=" * 60)
        logger.info("STRATEGY 2: SMOTE Oversampling")
        logger.info("=" * 60)
        
        # Apply SMOTE only to training data
        smote = SMOTE(random_state=self.random_state, sampling_strategy=0.5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE - Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        results = {}
        # Use simpler configs for SMOTE (already balanced)
        simple_models = {
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                random_state=self.random_state, eval_metric='logloss',
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=self.random_state, n_jobs=-1, verbose=-1
            )
        }
        
        for name, model in simple_models.items():
            logger.info(f"\nTraining {name} with SMOTE...")
            start_time = time.time()
            
            model.fit(X_resampled, y_resampled)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            result = {
                'model': model,
                'strategy': 'smote',
                'train_time': train_time,
                'val_precision': precision_score(y_val, y_pred, zero_division=0),
                'val_recall': recall_score(y_val, y_pred, zero_division=0),
                'val_f1': f1_score(y_val, y_pred, zero_division=0),
                'val_roc_auc': roc_auc_score(y_val, y_proba),
                'val_pr_auc': average_precision_score(y_val, y_proba),
            }
            
            results[f"{name} (SMOTE)"] = result
            self._log_result(f"{name} (SMOTE)", result)
            
            if result['val_f1'] > self.best_score:
                self.best_score = result['val_f1']
                self.best_model = model
                self.best_model_name = f"{name} (SMOTE)"
        
        self.results.update(results)
        return results
    
    def train_with_undersampling(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Strategy 3: Train models using random undersampling on training data only.
        """
        logger.info("=" * 60)
        logger.info("STRATEGY 3: Random Undersampling")
        logger.info("=" * 60)
        
        # Undersample majority class
        undersampler = RandomUnderSampler(
            random_state=self.random_state, 
            sampling_strategy=0.5  # 2:1 ratio instead of 1:1 to preserve some information
        )
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        
        logger.info(f"After undersampling - Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        results = {}
        simple_models = {
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                random_state=self.random_state, eval_metric='logloss',
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=self.random_state, n_jobs=-1, verbose=-1
            )
        }
        
        for name, model in simple_models.items():
            logger.info(f"\nTraining {name} with undersampling...")
            start_time = time.time()
            
            model.fit(X_resampled, y_resampled)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            result = {
                'model': model,
                'strategy': 'undersampling',
                'train_time': train_time,
                'val_precision': precision_score(y_val, y_pred, zero_division=0),
                'val_recall': recall_score(y_val, y_pred, zero_division=0),
                'val_f1': f1_score(y_val, y_pred, zero_division=0),
                'val_roc_auc': roc_auc_score(y_val, y_proba),
                'val_pr_auc': average_precision_score(y_val, y_proba),
            }
            
            results[f"{name} (Undersampled)"] = result
            self._log_result(f"{name} (Undersampled)", result)
            
            if result['val_f1'] > self.best_score:
                self.best_score = result['val_f1']
                self.best_model = model
                self.best_model_name = f"{name} (Undersampled)"
        
        self.results.update(results)
        return results
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all trained models."""
        rows = []
        for name, result in self.results.items():
            rows.append({
                'Model': name,
                'Strategy': result.get('strategy', 'N/A'),
                'Precision': round(result['val_precision'], 4),
                'Recall': round(result['val_recall'], 4),
                'F1-Score': round(result['val_f1'], 4),
                'ROC-AUC': round(result['val_roc_auc'], 4),
                'PR-AUC': round(result['val_pr_auc'], 4),
                'Train Time (s)': round(result['train_time'], 2)
            })
        
        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values('F1-Score', ascending=False).reset_index(drop=True)
        return comparison
    
    def save_best_model(self, filepath: str):
        """Save the best performing model."""
        if self.best_model is None:
            raise ValueError("No model trained yet. Run training first.")
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"Best model ({self.best_model_name}) saved to {filepath}")
    
    def save_all_results(self, directory: str):
        """Save all models and comparison results."""
        os.makedirs(directory, exist_ok=True)
        
        # Save best model
        self.save_best_model(os.path.join(directory, 'best_model.joblib'))
        
        # Save comparison table
        comparison = self.get_comparison_table()
        comparison.to_csv(os.path.join(directory, 'model_comparison.csv'), index=False)
        
        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_f1_score': self.best_score,
            'total_models_trained': len(self.results)
        }
        joblib.dump(metadata, os.path.join(directory, 'training_metadata.joblib'))
        
        logger.info(f"All results saved to {directory}")
    
    def _log_result(self, name: str, result: Dict):
        """Log a single model's results."""
        logger.info(f"  {name}:")
        logger.info(f"    Precision: {result['val_precision']:.4f}")
        logger.info(f"    Recall:    {result['val_recall']:.4f}")
        logger.info(f"    F1-Score:  {result['val_f1']:.4f}")
        logger.info(f"    ROC-AUC:   {result['val_roc_auc']:.4f}")
        logger.info(f"    PR-AUC:    {result['val_pr_auc']:.4f}")
        logger.info(f"    Time:      {result['train_time']:.2f}s")
    
    @staticmethod
    def _count_combinations(params: Dict) -> int:
        """Count total parameter combinations."""
        count = 1
        for values in params.values():
            count *= len(values)
        return count
    
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  n_iter: int = 20) -> pd.DataFrame:
        """
        Train all models with all imbalance-handling strategies.
        
        Returns comparison table.
        """
        logger.info("Starting comprehensive model training...")
        logger.info(f"Training data: {X_train.shape[0]} samples, {y_train.sum()} fraud")
        logger.info(f"Validation data: {X_val.shape[0]} samples, {y_val.sum()} fraud")
        
        # Strategy 1: Class weighting with hyperparameter tuning
        self.train_with_class_weight(X_train, y_train, X_val, y_val, n_iter=n_iter)
        
        # Strategy 2: SMOTE
        self.train_with_smote(X_train, y_train, X_val, y_val)
        
        # Strategy 3: Undersampling
        self.train_with_undersampling(X_train, y_train, X_val, y_val)
        
        comparison = self.get_comparison_table()
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\n{comparison.to_string()}")
        logger.info(f"\nBest model: {self.best_model_name} (F1: {self.best_score:.4f})")
        
        return comparison


if __name__ == '__main__':
    from data_preprocessing import DataPreprocessor, generate_synthetic_data
    from feature_engineering import FeatureEngineer
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'creditcard.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    # Generate synthetic data if needed
    if not os.path.exists(data_path):
        generate_synthetic_data(output_path=data_path)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.run_pipeline(data_path)
    
    # Feature engineering
    fe = FeatureEngineer()
    X_train = fe.fit_transform(X_train)
    X_val = fe.transform(X_val)
    
    # Train all models
    trainer = FraudModelTrainer()
    comparison = trainer.train_all(X_train, y_train, X_val, y_val, n_iter=3)
    
    # Save results
    trainer.save_all_results(models_dir)
    preprocessor.save_preprocessor(os.path.join(models_dir, 'preprocessor.joblib'))
    joblib.dump(fe, os.path.join(models_dir, 'feature_engineer.joblib'))
    
    print("\n[SUCCESS] Training complete!")
    print(f"Best model: {trainer.best_model_name}")
    print(f"Best F1 Score: {trainer.best_score:.4f}")
