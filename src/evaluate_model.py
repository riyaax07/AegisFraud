"""
Model Evaluation Module for Credit Card Fraud Detection
=======================================================
Comprehensive evaluation with visualizations and threshold analysis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudModelEvaluator:
    """
    Comprehensive model evaluation for fraud detection.
    
    Generates:
    - Confusion matrix visualization
    - ROC curve
    - Precision-Recall curve
    - Feature importance plot
    - Threshold analysis
    - Classification report
    - Business impact analysis
    """
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.evaluation_results = {}
    
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                 model_name: str = 'Best Model') -> Dict:
        """
        Run complete evaluation suite on the test set.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: Test labels
            model_name: Name for labeling plots
            
        Returns:
            Dictionary of all evaluation metrics
        """
        logger.info(f"Evaluating {model_name} on test set ({X_test.shape[0]} samples)")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        self.evaluation_results = metrics
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SET EVALUATION: {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Precision:  {metrics['precision']:.4f}")
        logger.info(f"Recall:     {metrics['recall']:.4f}")
        logger.info(f"F1-Score:   {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC:     {metrics['pr_auc']:.4f}")
        logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
        
        # Generate all plots
        self.plot_confusion_matrix(y_test, y_pred, model_name)
        self.plot_roc_curve(y_test, y_proba, model_name)
        self.plot_precision_recall_curve(y_test, y_proba, model_name)
        self.plot_feature_importance(model, X_test.columns.tolist(), model_name)
        self.plot_threshold_analysis(y_test, y_proba, model_name)
        self.generate_business_report(metrics)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Generate and save confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute values
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        axes[0].set_title(f'Confusion Matrix (Counts)\n{model_name}')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        axes[1].set_title(f'Confusion Matrix (Normalized)\n{model_name}')
        axes[1].set_ylabel('Actual')
        axes[1].set_xlabel('Predicted')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {filepath}")
    
    def plot_roc_curve(self, y_true, y_proba, model_name: str):
        """Generate and save ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#2196F3', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {filepath}")
    
    def plot_precision_recall_curve(self, y_true, y_proba, model_name: str):
        """Generate and save Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='#4CAF50', lw=2,
                label=f'{model_name} (AP = {pr_auc:.4f})')
        
        # Baseline: fraction of positive class
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='gray', linestyle='--', 
                   label=f'Baseline (fraud rate = {baseline:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'precision_recall_curve.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Precision-Recall curve saved to {filepath}")
    
    def plot_feature_importance(self, model, feature_names: list, model_name: str, 
                                 top_n: int = 20):
        """Generate and save feature importance plot."""
        importance = None
        
        # Try different ways to get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_type = 'Feature Importance (Gini/Gain)'
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            importance_type = 'Feature Importance (|Coefficient|)'
        else:
            logger.warning("Model does not have feature_importances_ or coef_")
            return
        
        # Create DataFrame and sort
        fi_df = pd.DataFrame({
            'Feature': feature_names[:len(importance)],
            'Importance': importance
        }).sort_values('Importance', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color='#FF9800')
        ax.set_xlabel(importance_type, fontsize=12)
        ax.set_title(f'Top {top_n} {importance_type}\n{model_name}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {filepath}")
    
    def plot_threshold_analysis(self, y_true, y_proba, model_name: str):
        """
        Analyze how different classification thresholds affect precision and recall.
        
        This is critical for fraud detection where the default 0.5 threshold 
        may not be optimal.
        """
        thresholds = np.arange(0.1, 0.95, 0.05)
        precisions = []
        recalls = []
        f1_scores_list = []
        
        for threshold in thresholds:
            y_pred_t = (y_proba >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
            recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
            f1_scores_list.append(f1_score(y_true, y_pred_t, zero_division=0))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, precisions, 'b-', label='Precision', lw=2)
        ax.plot(thresholds, recalls, 'r-', label='Recall', lw=2)
        ax.plot(thresholds, f1_scores_list, 'g--', label='F1-Score', lw=2)
        
        # Mark optimal F1 threshold
        best_idx = np.argmax(f1_scores_list)
        best_threshold = thresholds[best_idx]
        ax.axvline(x=best_threshold, color='gray', linestyle=':', alpha=0.7,
                   label=f'Optimal F1 threshold = {best_threshold:.2f}')
        
        ax.set_xlabel('Classification Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Threshold Analysis\n{model_name}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'threshold_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Threshold analysis saved to {filepath}")
        logger.info(f"Optimal threshold for F1: {best_threshold:.2f}")
        
        return best_threshold
    
    def generate_business_report(self, metrics: Dict):
        """
        Generate a business-oriented report explaining the model's performance
        in terms of real-world impact.
        """
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        total = tn + fp + fn + tp
        
        report = f"""
================================================================
           FRAUD DETECTION - BUSINESS IMPACT REPORT             
================================================================

  Model: {metrics['model_name']}

  DETECTION PERFORMANCE
  ---------------------
  Fraud cases correctly detected (TP):    {tp:>6,d}
  Fraud cases missed (FN):                {fn:>6,d}
  Legitimate flagged as fraud (FP):       {fp:>6,d}
  Legitimate correctly passed (TN):       {tn:>6,d}

  BUSINESS METRICS
  ----------------
  Fraud Detection Rate (Recall):   {metrics['recall']*100:>6.2f}%
  False Alarm Rate (FPR):          {fp/(fp+tn)*100:>6.2f}%
  Precision (of alerts):           {metrics['precision']*100:>6.2f}%

  TRADE-OFF ANALYSIS
  ------------------
  For every 100 fraud attempts:
    [DETECTED]  {metrics['recall']*100:>5.1f} are correctly detected and blocked
    [MISSED]    {(1-metrics['recall'])*100:>5.1f} slip through undetected

  For every 100 fraud alerts raised:
    [ACCURATE]  {metrics['precision']*100:>5.1f} are actual fraud cases
    [FALSE]     {(1-metrics['precision'])*100:>5.1f} are false alarms (legitimate transactions)

================================================================

BUSINESS INTERPRETATION:
1. Missed Fraud Cost: The model misses {fn} fraudulent transactions out of {tp+fn}.
   If average fraud amount is $500, potential undetected loss = ${fn * 500:,.0f}

2. Customer Impact: {fp} legitimate customers may experience blocked transactions
   or receive fraud alerts. This is {fp/(fp+tn)*100:.4f}% of all legitimate transactions.

3. Recommendation: {'High recall model - good for minimizing missed fraud.' if metrics['recall'] > 0.8 else 'Consider lowering the classification threshold to catch more fraud cases.'}
   {'False alarm rate is acceptable.' if fp/(fp+tn) < 0.01 else 'Consider raising threshold to reduce customer friction from false alarms.'}
"""
        
        filepath = os.path.join(self.output_dir, 'business_report.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Business report saved to {filepath}")
        print(report)


if __name__ == '__main__':
    from data_preprocessing import DataPreprocessor, generate_synthetic_data
    from feature_engineering import FeatureEngineer
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'creditcard.csv')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    
    if not os.path.exists(data_path):
        generate_synthetic_data(output_path=data_path)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.run_pipeline(data_path)
    
    # Load or fit feature engineer
    fe_path = os.path.join(models_dir, 'feature_engineer.joblib')
    if os.path.exists(fe_path):
        fe = joblib.load(fe_path)
    else:
        fe = FeatureEngineer()
        fe.fit(X_train)
    
    X_test_fe = fe.transform(X_test)
    
    # Load champion model
    best_model_path = os.path.join(models_dir, 'best_model.joblib')
    meta_path = os.path.join(models_dir, 'training_metadata.joblib')
    
    if os.path.exists(best_model_path):
        model = joblib.load(best_model_path)
        meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        model_name = meta.get('best_model_name', 'Random Forest (Undersampled)')
    else:
        from train_model import FraudModelTrainer
        trainer = FraudModelTrainer()
        trainer.train_all(fe.transform(X_train), y_train, fe.transform(X_val), y_val, n_iter=3)
        model = trainer.best_model
        model_name = trainer.best_model_name
    
    # Evaluate model on test set
    evaluator = FraudModelEvaluator(output_dir=reports_dir)
    metrics = evaluator.evaluate(model, X_test_fe, y_test, model_name)
    
    print("\n[SUCCESS] Evaluation complete! Generated visualizations in reports/ directory.")
