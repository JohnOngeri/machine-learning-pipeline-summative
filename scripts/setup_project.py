"""
🎤 DEEPFAKE VOICE DETECTION - PROJECT SETUP
Run this first to set up everything
"""

import os
import sys
import pandas as pd
sys.path.append('.')

from scripts.validate_audio_files import validate_audio_files
from scripts.process_raw_audio import process_audio_files
from scripts.train_initial_model import main as train_model
from src.preprocessing import AudioPreprocessor
from src.model import DeepfakeDetectionModel
from loguru import logger

def setup_project():
    """Complete project setup"""
    
    print("🎤 DEEPFAKE VOICE DETECTION - PROJECT SETUP")
    print("="*50)
    
    # Create folders
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Your 3 data sources
    audio_real_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\data\AUDIO\REAL"
    audio_fake_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\data\AUDIO\FAKE"
    csv_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\data\DATASET-balanced.csv"
    audio_base_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\data\AUDIO"
    
    preprocessor = AudioPreprocessor()
    final_dataset = None
    
    print("\n🔍 Step 1: Checking what data you have...")
    
    # Check what data exists
    has_audio = os.path.exists(audio_real_path) and os.path.exists(audio_fake_path)
    has_csv = os.path.exists(csv_path)
    
    if has_audio:
        real_files = len([f for f in os.listdir(audio_real_path) if f.endswith(('.wav', '.mp3', '.flac'))])
        fake_files = len([f for f in os.listdir(audio_fake_path) if f.endswith(('.wav', '.mp3', '.flac'))])
        print(f"✅ Found {real_files} REAL audio files")
        print(f"✅ Found {fake_files} FAKE audio files")
    else:
        print("❌ Audio files not found")
    
    if has_csv:
        csv_df = pd.read_csv(csv_path)
        print(f"✅ Found CSV with {len(csv_df)} samples")
    else:
        print("❌ CSV file not found")
    
    print("\n🎯 Step 2: Using the BEST strategy...")
    
    # OPTION 3: Use Both (Best of Both Worlds)
    if has_audio and has_csv:
        print("🌟 OPTION 3: Using BOTH audio files AND CSV (Best of Both Worlds!)")
        
        # Process audio files
        print("🔄 Processing your audio files...")
        audio_df = preprocessor.process_audio_directory(audio_base_path)
        
        # Load CSV data
        print("📊 Loading CSV data...")
        csv_df = pd.read_csv(csv_path)
        
        # Combine both datasets
        print("🔗 Combining both datasets...")
        if len(audio_df) > 0:
            # Remove filename column for combining
            if 'filename' in audio_df.columns:
                audio_df = audio_df.drop('filename', axis=1)
            if 'filename' in csv_df.columns:
                csv_df = csv_df.drop('filename', axis=1)
            
            # Combine datasets
            final_dataset = pd.concat([csv_df, audio_df], ignore_index=True)
            print(f"🎉 Combined dataset: {len(csv_df)} CSV + {len(audio_df)} audio = {len(final_dataset)} total!")
        else:
            print("⚠️  No audio processed, using CSV only")
            final_dataset = csv_df
            
    elif has_audio:
        print("🎵 Using audio files only...")
        final_dataset = preprocessor.process_audio_directory(audio_base_path)
        
    elif has_csv:
        print("📊 Using CSV file only...")
        final_dataset = pd.read_csv(csv_path)
        
    else:
        print("❌ No data found! Please check your file paths.")
        return
    
    # Save the combined dataset
    final_dataset.to_csv('data/DATASET-balanced.csv', index=False)
    print(f"💾 Saved combined dataset: {len(final_dataset)} samples")
    
    print("\n🤖 Step 3: Training initial model...")
    try:
        train_model()
        print("✅ Model training completed!")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        print("❌ Model training failed!")
        return
    
    # Step 4: Final setup
    print("\n🎯 Step 4: Final setup...")
    
    # Create additional directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    os.makedirs('data/retrain/REAL', exist_ok=True)
    os.makedirs('data/retrain/FAKE', exist_ok=True)
    
    print("✅ Project setup completed!")
    
    print("\n🎉 SUCCESS! Your voice detective is ready!")
    print("="*50)
    print(f"📊 Total training data: {len(final_dataset)} samples")
    print("\n🚀 READY TO START!")
    print("="*50)
    print("Your deepfake detection system is ready!")
    print("\nNext steps:")
    print("1. Start API server:")
    print("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("\n2. Start web interface:")
    print("   streamlit run ui/app.py")
    print("\n3. Or use Docker:")
    print("   docker-compose up --build")
    print("\n4. Access web interface: http://localhost:8501")
    print("5. Access API docs: http://localhost:8000/docs")
    print("="*50)

if __name__ == "__main__":
    setup_project()
