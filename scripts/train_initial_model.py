"""
Script to train the initial model from the dataset
Run this script to create the initial model files
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
    logger.info("Starting initial model training...")

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
    dataset_path = os.path.join(data_dir, "DATASET-balanced.csv")
    model_path = os.path.join(model_dir, "best_model.pkl")
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")

    # Download and save dataset
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/DATASET-balanced-JcqFJYhgnWK5P8zrmIuuMwyj9BIpH9.csv"
    logger.info("Downloading dataset...")
    df = pd.read_csv(url)

    df.to_csv(dataset_path, index=False)
    logger.info(f"Dataset saved to {dataset_path}")

    # Initialize preprocessor and prepare data
    preprocessor = AudioPreprocessor()
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

    # Extract metrics
    metrics = evaluation_results.get("metrics", {})

    # Print metrics for verification
    print("\nEvaluation metrics returned:")
    import pprint
    pprint.pprint(metrics)
    print()

    # Ensure training_history is initialized
    if not hasattr(model, "training_history") or not isinstance(model.training_history, dict):
        model.training_history = {}

    # Merge evaluation metrics into training history
    model.training_history.update({
        "val_accuracy": metrics.get("accuracy", 0),
        "f1_score": metrics.get("f1", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0)
    })

    # Optionally store full evaluation metrics
    model.evaluation_metrics = metrics

    # Log final history
    logger.info(f"Final training history: {model.training_history}")

    # Save model and preprocessor
    model.save_model(model_path)
    preprocessor.save_preprocessor(preprocessor_path)


    # Print summary
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
    print(f"- {model_path}")
    print(f"- {preprocessor_path}")
    print(f"- {dataset_path}")
    print("\nYou can now start the API server:")
    print("uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("="*50)

if __name__ == "__main__":
    main()
