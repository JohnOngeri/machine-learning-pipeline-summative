"""
Script to load pre-saved model and preprocessor
Run this script to use the pre-saved model and preprocessor
"""

import pandas as pd
import os
import sys
sys.path.append('.')
sys.path.insert(0, '..')

from src.preprocessing import AudioPreprocessor
from src.model import DeepfakeDetectionModel
from loguru import logger

def main():
    logger.info("Loading pre-saved model and preprocessor...")

    # Define base project path
    base_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative"
    data_dir = os.path.join(base_path, "data")
    model_dir = os.path.join(base_path, "models")
    log_dir = os.path.join(base_path, "logs")

    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Define absolute file paths
    # Explicit absolute path for the dataset
    dataset_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\data\DATASET-balanced.csv"

    model_path = os.path.join(model_dir, "best_model.pkl")
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")

    # Load pre-saved model
    model = DeepfakeDetectionModel()
    model.load_model(model_path)

    # Load pre-saved preprocessor
    preprocessor = AudioPreprocessor()
    preprocessor.load_preprocessor(preprocessor_path)

    # Print summary
    logger.info("Pre-saved model and preprocessor loaded!")
    print("\n" + "="*50)
    print("PRE-SAVED MODEL AND PREPROCESSOR LOADED")
    print("="*50)
    
    print(f"Preprocessor Type: {preprocessor.__class__.__name__}")
    print("\nFiles loaded:")
    print(f"- {model_path}")
    print(f"- {preprocessor_path}")
    print("\nYou can now start the API server:")
    print("uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("="*50)

if __name__ == "__main__":
    main()