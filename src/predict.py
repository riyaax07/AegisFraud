"""
Prediction Module for Credit Card Fraud Detection
=================================================
Provides prediction and explanation capabilities for individual transactions.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Production-ready fraud prediction with explainability.
    
    Loads a trained model and preprocessor, then provides:
    - Fraud probability prediction
    - Risk level classification
    - Feature-based explanation of why a transaction was flagged
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the predictor with saved model artifacts.
        
        Args:
            model_dir: Directory containing saved model, preprocessor, and feature engineer
        """
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self.model_dir = os.path.join(base_dir, 'models')
        else:
            self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.metadata = None
        self._loaded = False
    
    def load(self):
        """Load all saved model artifacts."""
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'best_model.joblib')
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.joblib')
            if os.path.exists(preprocessor_path):
                from data_preprocessing import DataPreprocessor
                self.preprocessor = DataPreprocessor.load_preprocessor(preprocessor_path)
                logger.info("Preprocessor loaded")
            
            # Load feature engineer
            fe_path = os.path.join(self.model_dir, 'feature_engineer.joblib')
            if os.path.exists(fe_path):
                self.feature_engineer = joblib.load(fe_path)
                logger.info("Feature engineer loaded")
            
            # Load metadata
            meta_path = os.path.join(self.model_dir, 'training_metadata.joblib')
            if os.path.exists(meta_path):
                self.metadata = joblib.load(meta_path)
                logger.info(f"Metadata loaded: {self.metadata}")
            
            self._loaded = True
            logger.info("All artifacts loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Model artifacts not found: {e}")
            raise
    
    def predict(self, transaction: Dict) -> Dict:
        """
        Predict whether a single transaction is fraudulent.
        
        Args:
            transaction: Dictionary with transaction features
                Required keys depend on the model, but typically:
                - V1 through V28: PCA-transformed features
                - Amount: Transaction amount
                - Time: Time elapsed (optional)
                
        Returns:
            Dictionary with prediction results:
            - is_fraud: Boolean prediction
            - fraud_probability: Float probability [0, 1]
            - risk_level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
            - confidence: Float confidence score
            - explanation: List of top factors
        """
        if not self._loaded:
            self.load()
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Ensure required columns exist
        feature_cols = self._get_expected_features()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0  # Default value for missing features
        
        # Apply preprocessing
        if self.preprocessor and self.preprocessor._is_fitted:
            df = self.preprocessor.transform(df)
        
        # Apply feature engineering
        if self.feature_engineer:
            df = self.feature_engineer.transform(df)
        
        # Ensure columns match model's expected features
        if hasattr(self.model, 'feature_names_in_'):
            expected = list(self.model.feature_names_in_)
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[expected]
        
        # Get prediction
        prediction = self.model.predict(df)[0]
        proba = self.model.predict_proba(df)[0] if hasattr(self.model, 'predict_proba') else [1 - prediction, prediction]
        fraud_probability = float(proba[1])
        
        # Determine risk level
        risk_level = self._get_risk_level(fraud_probability)
        
        # Generate explanation
        explanation = self._explain_prediction(df, fraud_probability)
        
        result = {
            'is_fraud': bool(prediction),
            'fraud_probability': round(fraud_probability, 6),
            'risk_level': risk_level,
            'confidence': round(max(proba), 4),
            'explanation': explanation,
            'model_used': self.metadata.get('best_model_name', 'Unknown') if self.metadata else 'Unknown'
        }
        
        return result
    
    def predict_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for a batch of transactions.
        
        Args:
            transactions: DataFrame with transaction features
            
        Returns:
            DataFrame with original data plus prediction columns
        """
        if not self._loaded:
            self.load()
        
        results = []
        for idx, row in transactions.iterrows():
            result = self.predict(row.to_dict())
            results.append(result)
        
        results_df = pd.DataFrame(results)
        return pd.concat([transactions.reset_index(drop=True), results_df], axis=1)
    
    def _get_risk_level(self, probability: float) -> str:
        """Classify risk level based on fraud probability."""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.6:
            return 'MEDIUM'
        elif probability < 0.85:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _explain_prediction(self, X: pd.DataFrame, fraud_probability: float) -> List[Dict]:
        """
        Generate human-readable explanations for the prediction.
        
        Uses feature importance to identify the top factors.
        """
        explanations = []
        
        # Get feature importance
        importance = None
        feature_names = X.columns.tolist()
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            importance = np.abs(coef)
        
        if importance is not None and len(importance) == len(feature_names):
            # Get top contributing features
            feature_contributions = []
            for i, (name, imp) in enumerate(zip(feature_names, importance)):
                if i < len(X.columns):
                    value = float(X.iloc[0, i])
                    feature_contributions.append({
                        'feature': name,
                        'importance': float(imp),
                        'value': round(value, 4),
                        'direction': 'increases fraud risk' if value * imp > 0 else 'decreases fraud risk'
                    })
            
            # Sort by importance and take top 5
            feature_contributions.sort(key=lambda x: x['importance'], reverse=True)
            explanations = feature_contributions[:5]
        
        return explanations
    
    def _get_expected_features(self) -> List[str]:
        """Get the list of features the model expects."""
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        
        # Default Kaggle ULB features
        base_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        return base_features
    
    def get_sample_transactions(self) -> List[Dict]:
        """
        Return sample transactions for testing the prediction API.
        """
        samples = [
            {
                'name': 'Normal Low-Amount Transaction',
                'transaction': {
                    'Time': 50000,
                    'Amount': 12.50,
                    **{f'V{i}': np.random.normal(0, 0.5) for i in range(1, 29)}
                }
            },
            {
                'name': 'Suspicious High-Amount Transaction',
                'transaction': {
                    'Time': 80000,
                    'Amount': 987.65,
                    'V1': -3.5, 'V2': 2.1, 'V3': -4.2, 'V4': 2.8,
                    'V5': -0.5, 'V6': -1.2, 'V7': -3.1, 'V8': 0.5,
                    'V9': -2.0, 'V10': -5.1, 'V11': 1.5, 'V12': 3.2,
                    'V13': -0.8, 'V14': -4.5, 'V15': 0.3, 'V16': -2.5,
                    'V17': -4.0, 'V18': -1.0, 'V19': 0.4, 'V20': 0.1,
                    'V21': -0.5, 'V22': -0.2, 'V23': -0.1, 'V24': 0.3,
                    'V25': 0.1, 'V26': -0.3, 'V27': 0.2, 'V28': 0.1
                }
            },
            {
                'name': 'Normal Medium-Amount Transaction',
                'transaction': {
                    'Time': 120000,
                    'Amount': 75.00,
                    **{f'V{i}': np.random.normal(0, 0.3) for i in range(1, 29)}
                }
            }
        ]
        return samples


if __name__ == '__main__':
    import sys
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, 'models')
    
    # Add src to path for imports
    sys.path.insert(0, os.path.dirname(__file__))
    
    predictor = FraudPredictor(model_dir=model_dir)
    
    try:
        predictor.load()
    except FileNotFoundError:
        print("Model not found. Please run train_model.py first.")
        sys.exit(1)
    
    # Test with sample transactions
    samples = predictor.get_sample_transactions()
    
    for sample in samples:
        print(f"\n{'='*50}")
        print(f"Transaction: {sample['name']}")
        print(f"{'='*50}")
        
        result = predictor.predict(sample['transaction'])
        
        print(f"  Fraud: {result['is_fraud']}")
        print(f"  Probability: {result['fraud_probability']:.6f}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Model: {result['model_used']}")
        
        if result['explanation']:
            print(f"  Top factors:")
            for exp in result['explanation']:
                print(f"    - {exp['feature']}: {exp['value']} ({exp['direction']})")
