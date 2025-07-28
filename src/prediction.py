import numpy as np
import pandas as pd
from typing import Dict, Any, List
from loguru import logger
import os
import sys
sys.path.insert(0, '..')
from preprocessing import AudioPreprocessor

from model import DeepfakeDetectionModel

import sys
sys.path.insert(0, '..')
print(sys.path)
class DeepfakePredictionService:
    """Service for making predictions on audio files"""
    
    def __init__(self, 
                 model_path: str = None, 
                 preprocessor_path: str = None):

        base_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative"
        self.model_path = model_path or os.path.join(base_path, "models", "best_model.pkl")
        self.preprocessor_path = preprocessor_path or os.path.join(base_path, "models", "preprocessor.pkl")
        
        self.model = None
        self.preprocessor = None
        self.load_components()
    
    def load_components(self):
        """Load model and preprocessor"""
        try:
            # Load preprocessor
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = AudioPreprocessor()
                self.preprocessor.load_preprocessor(self.preprocessor_path)
                logger.info("Preprocessor loaded successfully")
            else:
                logger.warning(f"Preprocessor not found at {self.preprocessor_path}")
            
            # Load model
            if os.path.exists(self.model_path):
                self.model = DeepfakeDetectionModel()
                self.model.load_model(self.model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {self.model_path}")
                
        except Exception as e:
            logger.error(f"Error loading components: {e}")
    
    def predict_single_file(self, audio_path: str) -> Dict[str, Any]:
        """Predict if a single audio file is real or fake"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model or preprocessor not loaded")
        
        try:
            # Preprocess the audio file
            X = self.preprocessor.preprocess_single_file(audio_path)
            
            # Make prediction
            predictions, probabilities = self.model.predict(X)
            
            # Convert prediction to label
            label = self.preprocessor.label_encoder.inverse_transform(predictions)[0]
            confidence = np.max(probabilities[0])
            
            # Get probabilities for both classes
            prob_fake = probabilities[0][0] if self.preprocessor.label_encoder.classes_[0] == 'FAKE' else probabilities[0][1]
            prob_real = probabilities[0][1] if self.preprocessor.label_encoder.classes_[1] == 'REAL' else probabilities[0][0]
            
            result = {
                'filename': os.path.basename(audio_path),
                'prediction': label,
                'confidence': float(confidence),
                'probabilities': {
                    'FAKE': float(prob_fake),
                    'REAL': float(prob_real)
                },
                'status': 'success'
            }
            
            logger.info(f"Prediction for {audio_path}: {label} (confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting {audio_path}: {e}")
            return {
                'filename': os.path.basename(audio_path),
                'prediction': None,
                'confidence': None,
                'probabilities': None,
                'status': 'error',
                'error': str(e)
            }
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict multiple audio files"""
        results = []
        for audio_path in audio_paths:
            result = self.predict_single_file(audio_path)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {'status': 'Model not loaded'}
        
        info = {
            'model_type': self.model.model_type,
            'model_loaded': True,
            'preprocessor_loaded': self.preprocessor is not None,
            'training_history': self.model.training_history,
            'best_params': self.model.best_params
        }
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the service is ready"""
        return {
            'model_loaded': self.model is not None,
            'preprocessor_loaded': self.preprocessor is not None,
            'model_path_exists': os.path.exists(self.model_path),
            'preprocessor_path_exists': os.path.exists(self.preprocessor_path),
            'status': 'healthy' if (self.model is not None and self.preprocessor is not None) else 'unhealthy'
        }
