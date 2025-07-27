"""
Script to train the initial model from the dataset
Run this script to create the initial model files
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')

from src.preprocessing import AudioPreprocessor
from src.model import DeepfakeDetectionModel
from loguru import logger

def main():
    logger.info("Starting initial model training...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Download and prepare dataset
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/DATASET-balanced-JcqFJYhgnWK5P8zrmIuuMwyj9BIpH9.csv"
    
    logger.info("Downloading dataset...")
    df = pd.read_csv(url)
    
    # Save dataset locally
    dataset_path = 'data/DATASET-balanced.csv'
    df.to_csv(dataset_path, index=False)
    logger.info(f"Dataset saved to {dataset_path}")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Load and prepare data
    logger.info("Preparing data...")
    X, y = preprocessor.load_and_prepare_data(csv_path=dataset_path)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train model
    logger.info("Training Random Forest model...")
    model = DeepfakeDetectionModel(model_type='random_forest')
    training_history = model.train(X_train, y_train, X_test, y_test, use_grid_search=True)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluation_results = model.evaluate(X_test, y_test)
    
    # Save model and preprocessor
    model.save_model('models/best_model.pkl')
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Print results
    logger.info("Training completed!")
    logger.info(f"Cross-validation F1 Score: {training_history['cv_mean']:.4f}")
    logger.info(f"Test Accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {evaluation_results['metrics']['f1']:.4f}")
    
    print("\n" + "="*50)
    print("INITIAL MODEL TRAINING COMPLETED")
    print("="*50)
    print(f"Dataset: {len(df)} samples")
    print(f"Features: {len(preprocessor.feature_columns)}")
    print(f"Model Type: Random Forest")
    print(f"CV F1 Score: {training_history['cv_mean']:.4f} ± {training_history['cv_std']:.4f}")
    print(f"Test Accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
    print(f"Test F1 Score: {evaluation_results['metrics']['f1']:.4f}")
    print("\nFiles created:")
    print("- models/best_model.pkl")
    print("- models/preprocessor.pkl")
    print("- data/DATASET-balanced.csv")
    print("\nYou can now start the API server:")
    print("uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("="*50)

if __name__ == "__main__":
    main()
