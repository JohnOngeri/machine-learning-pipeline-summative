import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from loguru import logger
import os

class DeepfakeDetectionModel:
    """Machine Learning model for deepfake voice detection"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.training_history = {}
        
    def _get_model(self, **params):
        """Get model instance based on type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=42
            )
        elif self.model_type == 'svm':
            return SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              use_grid_search: bool = True) -> Dict[str, Any]:
        """Train the model with optional hyperparameter tuning"""
        
        logger.info(f"Training {self.model_type} model...")
        
        if use_grid_search:
            logger.info("Performing hyperparameter tuning...")
            param_grid = self._get_param_grid()
            
            base_model = self._get_model()
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=5, 
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {self.best_params}")
        else:
            self.model = self._get_model()
            self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        
        self.training_history = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': self.best_params
        }
        
        logger.info(f"Training completed. CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.training_history
    
    def _get_param_grid(self) -> Dict[str, Any]:
        """Get parameter grid for hyperparameter tuning"""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'logistic_regression':
            return {
                'C': [0.1, 1.0, 10.0, 100.0],
                'max_iter': [1000, 2000]
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear']
            }
        else:
            return {}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
        
        return evaluation_results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def save_model(self, path: str = 'models/best_model.pkl'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = 'models/best_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.best_params = model_data.get('best_params')
        self.training_history = model_data.get('training_history', {})
        logger.info(f"Model loaded from {path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: list = ['FAKE', 'REAL'], 
                            save_path: str = None) -> plt.Figure:
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (for tree-based models)"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            logger.warning("Model does not support feature importance")
            return None
