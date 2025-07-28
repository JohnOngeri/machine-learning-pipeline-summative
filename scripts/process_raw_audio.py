"""
Script to process raw audio files and create dataset
Use this to process your audio files from the data/AUDIO directory
"""

import os
import sys
sys.path.append('.')
import sys
sys.path.insert(0, '..')

from src.preprocessing import AudioPreprocessor
from loguru import logger
import pandas as pd

def process_audio_files():
    """Process raw audio files and create CSV dataset"""
    
    # Your audio file paths
    audio_base_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\data\AUDIO"
    
    logger.info(f"Processing audio files from: {audio_base_path}")
    
    # Check if directories exist
    real_dir = os.path.join(audio_base_path, "REAL")
    fake_dir = os.path.join(audio_base_path, "FAKE")
    
    if not os.path.exists(real_dir):
        logger.error(f"REAL directory not found: {real_dir}")
        return
    
    if not os.path.exists(fake_dir):
        logger.error(f"FAKE directory not found: {fake_dir}")
        return
    
    # Count files
    real_files = [f for f in os.listdir(real_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    fake_files = [f for f in os.listdir(fake_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    logger.info(f"Found {len(real_files)} REAL audio files")
    logger.info(f"Found {len(fake_files)} FAKE audio files")
    
    if len(real_files) == 0 and len(fake_files) == 0:
        logger.error("No audio files found!")
        return
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Process audio files
    logger.info("Starting feature extraction...")
    df = preprocessor.process_audio_directory(audio_base_path)
    
    if len(df) == 0:
        logger.error("No features extracted from audio files!")
        return
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save processed dataset
    output_path = 'data/DATASET-balanced.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"Dataset created successfully!")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"REAL samples: {len(df[df['LABEL'] == 'REAL'])}")
    logger.info(f"FAKE samples: {len(df[df['LABEL'] == 'FAKE'])}")
    logger.info(f"Features extracted: {len([col for col in df.columns if col not in ['LABEL', 'filename']])}")
    logger.info(f"Dataset saved to: {output_path}")
    
    # Display sample
    print("\nSample of processed data:")
    print(df.head())
    
    print("\nDataset statistics:")
    print(df.describe())

if __name__ == "__main__":
    process_audio_files()
