import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import os
import json
class DeepfakeDetectionModel:
    """Pretrained Deepfake Voice Detection Model Wrapper"""

    def __init__(self):
        self.model = None
        self.preprocessor = None

    def load_model(self, path: str = r'C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\models\best_model.pkl'):
        """Load the trained ML model"""
        print(f"[INFO] Loading model from: {path}")
        self.model = joblib.load(path)
        print(f"[SUCCESS] Model loaded: {type(self.model)}")

    def load_preprocessor(self, path: str = r'C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\models\preprocessor.pkl'):
        """Load the saved preprocessor"""
        print(f"[INFO] Loading preprocessor from: {path}")
        self.preprocessor = joblib.load(path)
        print(f"[SUCCESS] Preprocessor loaded: {type(self.preprocessor)}")

    def predict_proba(self, X):
        """Expose probability predictions externally"""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model.predict_proba(X)


    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new preprocessed data"""
        if self.model is None:
            raise ValueError("Model not loaded")
        if X is None:
            raise ValueError("Input features X is None")

        print(f"[MODEL] Predicting on input of shape: {X.shape}")
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            return predictions, probabilities
        except Exception as e:
            print(f"[MODEL ERROR] Prediction failed: {e}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not loaded")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }

    def plot_confusion_matrix(self, cm: np.ndarray, labels: list = ['FAKE', 'REAL'], save_path: str = None) -> plt.Figure:
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
    def training_history(self):
            """Load training metrics from saved JSON file"""
            metrics_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\models\evaluation_results_20250801_105126.json"
            if not os.path.exists(metrics_path):
                print(f"[WARNING] Metrics file not found at: {metrics_path}")
                return {}

            try:
                with open(metrics_path, "r") as f:
                    history = json.load(f)
                    print(f"[INFO] Loaded training history.")
                    return history
            except Exception as e:
                print(f"[ERROR] Failed to load training history: {e}")
                return {}
