import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from loguru import logger
from fastapi.encoders import jsonable_encoder
import json
import pprint

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import AudioPreprocessor
from model import DeepfakeDetectionModel


class DeepfakePredictionService:
    """Service for loading model/preprocessor and making predictions on audio files."""

    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        base_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative"
        self.model_path = model_path or os.path.join(base_path, "models", "best_model.pkl")
        self.preprocessor_path = preprocessor_path or os.path.join(base_path, "models", "preprocessor.pkl")
        
        self.model = None
        self.preprocessor = None
        self.load_components()

    def load_components(self):
        """Load the model and preprocessor from disk."""
        try:
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = AudioPreprocessor()
                self.preprocessor.load_preprocessor(self.preprocessor_path)
                logger.info("Preprocessor loaded successfully.")
            else:
                logger.warning(f"Preprocessor not found at: {self.preprocessor_path}")

            if os.path.exists(self.model_path):
                self.model = DeepfakeDetectionModel()
                self.model.load_model(self.model_path)
                logger.info("Model loaded successfully.")
            else:
                logger.warning(f"Model not found at: {self.model_path}")

        except Exception as e:
            logger.error(f"Error during component loading: {e}")

    def predict_single_file(self, audio_path: str) -> Dict[str, Any]:
        """Run prediction on a single audio file."""
        if not self.model or not self.preprocessor:
            raise ValueError("Model or preprocessor not loaded.")

        try:
            X = self.preprocessor.preprocess_single_file(audio_path)
            predictions, probabilities = self.model.predict(X)
            
            label = self.preprocessor.label_encoder.inverse_transform(predictions)[0]
            confidence = float(np.max(probabilities[0]))
            classes = self.preprocessor.label_encoder.classes_

            result = {
                'filename': os.path.basename(audio_path),
                'prediction': label,
                'confidence': confidence,
                'probabilities': {
                    'FAKE': float(probabilities[0][0]) if classes[0] == 'FAKE' else float(probabilities[0][1]),
                    'REAL': float(probabilities[0][1]) if classes[1] == 'REAL' else float(probabilities[0][0])
                },
                'status': 'success'
            }

            logger.info(f"Prediction for {audio_path}: {label} (confidence: {confidence:.4f})")
            return result

        except Exception as e:
            logger.error(f"Prediction error for {audio_path}: {e}")
            return {
                'filename': os.path.basename(audio_path),
                'prediction': None,
                'confidence': None,
                'probabilities': None,
                'status': 'error',
                'error': str(e)
            }

    def predict_batch(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict multiple audio files in a batch."""
        return [self.predict_single_file(path) for path in audio_paths]

    

  
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model, with safe serialization."""

        if self.model is None:
            logger.debug("Model is not loaded.")
            return {'status': 'Model not loaded'}

        logger.debug("Model is loaded.")

        # === Handle training_history ===
        training_history = self.model.training_history
        try:
            if hasattr(training_history, 'to_dict'):
                training_history = training_history.to_dict()
                logger.debug("Converted training_history from DataFrame to dict.")

            # Safely convert numpy types to JSON-serializable values
            def make_json_safe(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                return obj

            training_history = make_json_safe(training_history)
            logger.debug("Converted training_history to JSON-safe format.")
            print("\n=== Training History ===")
            pprint.pprint(training_history)

        except Exception as e:
            logger.warning(f"Failed to prepare training_history: {e}")
            training_history = {}

        # === Handle best_params ===
        try:
            best_params = jsonable_encoder(self.model.best_params)
            logger.debug("Encoded best_params to JSON-compatible format.")
            print("\n=== Best Params ===")
            pprint.pprint(best_params)
        except Exception as e:
            logger.warning(f"Failed to encode best_params: {e}")
            best_params = {}

        # === Construct response ===
        info = {
            'model_type': getattr(self.model, 'model_type', 'unknown'),
            'model_loaded': True,
            'preprocessor_loaded': self.preprocessor is not None,
            'training_history': training_history,
            'best_params': best_params
        }

        logger.debug("Model info successfully constructed.")
        print("\n=== Final Model Info ===")
        pprint.pprint(info)

        return info





    def health_check(self) -> Dict[str, Any]:
        """Perform a health check to verify if model and preprocessor are ready."""
        model_loaded = self.model is not None
        preprocessor_loaded = self.preprocessor is not None
        model_path_exists = os.path.exists(self.model_path)
        preprocessor_path_exists = os.path.exists(self.preprocessor_path)
        status = 'healthy' if (model_loaded and preprocessor_loaded) else 'unhealthy'

        logger.debug(f"Health Check - model_loaded: {model_loaded}, preprocessor_loaded: {preprocessor_loaded}")
        logger.debug(f"Model path exists: {model_path_exists} -> {self.model_path}")
        logger.debug(f"Preprocessor path exists: {preprocessor_path_exists} -> {self.preprocessor_path}")
        logger.debug(f"System status: {status}")

        return {
            'model_loaded': model_loaded,
            'preprocessor_loaded': preprocessor_loaded,
            'model_path_exists': model_path_exists,
            'preprocessor_path_exists': preprocessor_path_exists,
            'status': status
        }
