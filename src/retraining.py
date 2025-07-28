import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from loguru import logger
from datetime import datetime
import shutil
from preprocessing import AudioPreprocessor

from model import DeepfakeDetectionModel
class RetrainingService:
    """Service for retraining the model with new data"""
    
    def __init__(self, data_dir: str = 'data', models_dir: str = 'models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.retrain_dir = os.path.join(data_dir, 'retrain')
        self.backup_dir = os.path.join(models_dir, 'backups')
        
        # Create directories if they don't exist
        os.makedirs(self.retrain_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(os.path.join(self.retrain_dir, 'REAL'), exist_ok=True)
        os.makedirs(os.path.join(self.retrain_dir, 'FAKE'), exist_ok=True)
    
    def add_training_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add new audio files for retraining"""
        added_files = {'REAL': [], 'FAKE': []}
        errors = []
        
        for file_info in files:
            try:
                file_path = file_info['path']
                label = file_info['label'].upper()
                
                if label not in ['REAL', 'FAKE']:
                    errors.append(f"Invalid label {label} for file {file_path}")
                    continue
                
                # Copy file to retrain directory
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.retrain_dir, label, filename)
                
                shutil.copy2(file_path, dest_path)
                added_files[label].append(filename)
                
                logger.info(f"Added {filename} to {label} retraining set")
                
            except Exception as e:
                errors.append(f"Error processing {file_info.get('path', 'unknown')}: {str(e)}")
        
        return {
            'added_files': added_files,
            'errors': errors,
            'total_added': len(added_files['REAL']) + len(added_files['FAKE'])
        }
    
    def backup_current_model(self) -> str:
        """Backup current model before retraining"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"model_backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        os.makedirs(backup_path, exist_ok=True)
        
        # Backup model files
        model_files = ['best_model.pkl', 'preprocessor.pkl']
        for file in model_files:
            src_path = os.path.join(self.models_dir, file)
            if os.path.exists(src_path):
                dst_path = os.path.join(backup_path, file)
                shutil.copy2(src_path, dst_path)
        
        logger.info(f"Model backed up to {backup_path}")
        return backup_path
    
    def retrain_model(self, include_original_data: bool = True, 
                     model_type: str = 'random_forest') -> Dict[str, Any]:
        """Retrain the model with new data"""
        try:
            logger.info("Starting model retraining...")
            
            # Backup current model
            backup_path = self.backup_current_model()
            
            # Initialize preprocessor
            preprocessor = AudioPreprocessor()
            
            # Process new training data
            new_data = preprocessor.process_audio_directory(self.retrain_dir)
            
            if len(new_data) == 0:
                return {
                    'status': 'error',
                    'message': 'No new training data found',
                    'backup_path': backup_path
                }
            
            # Combine with original data if requested
            if include_original_data:
                original_csv = os.path.join(self.data_dir, 'DATASET-balanced.csv')
                if os.path.exists(original_csv):
                    original_data = pd.read_csv(original_csv)
                    # Remove filename column if it exists
                    if 'filename' in original_data.columns:
                        original_data = original_data.drop('filename', axis=1)
                    if 'filename' in new_data.columns:
                        new_data = new_data.drop('filename', axis=1)
                    
                    combined_data = pd.concat([original_data, new_data], ignore_index=True)
                    logger.info(f"Combined {len(original_data)} original + {len(new_data)} new samples")
                else:
                    combined_data = new_data
                    logger.info(f"Using only new data: {len(new_data)} samples")
            else:
                combined_data = new_data
                logger.info(f"Using only new data: {len(new_data)} samples")
            
            # Save updated dataset
            updated_csv_path = os.path.join(self.data_dir, 'DATASET-balanced.csv')
            combined_data.to_csv(updated_csv_path, index=False)
            
            # Prepare data for training
            X, y = preprocessor.load_and_prepare_data(csv_path=updated_csv_path)
            X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
            
            # Initialize and train model
            model = DeepfakeDetectionModel(model_type=model_type)
            training_history = model.train(X_train, y_train, X_test, y_test, use_grid_search=True)
            
            # Evaluate model
            evaluation_results = model.evaluate(X_test, y_test)
            
            # Save updated model and preprocessor
            model.save_model(os.path.join(self.models_dir, 'best_model.pkl'))
            preprocessor.save_preprocessor(os.path.join(self.models_dir, 'preprocessor.pkl'))
            
            # Clear retrain directory
            self._clear_retrain_directory()
            
            # Log retraining event
            self._log_retraining_event(training_history, evaluation_results, len(new_data))
            
            logger.info("Model retraining completed successfully")
            
            return {
                'status': 'success',
                'message': 'Model retrained successfully',
                'training_history': training_history,
                'evaluation_results': evaluation_results,
                'new_samples_count': len(new_data),
                'total_samples_count': len(combined_data),
                'backup_path': backup_path
            }
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return {
                'status': 'error',
                'message': f'Retraining failed: {str(e)}',
                'backup_path': backup_path if 'backup_path' in locals() else None
            }
    
    def _clear_retrain_directory(self):
        """Clear the retrain directory after successful retraining"""
        for label in ['REAL', 'FAKE']:
            label_dir = os.path.join(self.retrain_dir, label)
            if os.path.exists(label_dir):
                for file in os.listdir(label_dir):
                    os.remove(os.path.join(label_dir, file))
        logger.info("Retrain directory cleared")
    
    def _log_retraining_event(self, training_history: Dict, evaluation_results: Dict, new_samples: int):
        """Log retraining event details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'new_samples': new_samples,
            'cv_f1_score': training_history.get('cv_mean', 0),
            'test_accuracy': evaluation_results['metrics']['accuracy'],
            'test_f1_score': evaluation_results['metrics']['f1']
        }
        
        log_file = os.path.join('logs', 'retraining_log.txt')
        os.makedirs('logs', exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(f"{log_entry}\n")
    
    def get_retrain_status(self) -> Dict[str, Any]:
        """Get current retraining status"""
        real_files = len(os.listdir(os.path.join(self.retrain_dir, 'REAL')))
        fake_files = len(os.listdir(os.path.join(self.retrain_dir, 'FAKE')))
        
        return {
            'pending_files': {
                'REAL': real_files,
                'FAKE': fake_files,
                'total': real_files + fake_files
            },
            'ready_for_retraining': (real_files + fake_files) > 0
        }
