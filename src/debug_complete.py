import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from loguru import logger

# Your exact paths
BASE_PATH = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative"
PREPROCESSOR_PATH = os.path.join(BASE_PATH, "models", "preprocessor.pkl")
MODEL_PATH = os.path.join(BASE_PATH, "models", "best_model.pkl")
FAKE_AUDIO_PATH = os.path.join(BASE_PATH, "data", "AUDIO", "fake")
REAL_AUDIO_PATH = os.path.join(BASE_PATH, "data", "AUDIO", "real")

def debug_paths():
    """Check if all required paths exist"""
    print("=== DEBUGGING PATHS ===")
    
    paths_to_check = {
        "Base Path": BASE_PATH,
        "Preprocessor": PREPROCESSOR_PATH,
        "Model": MODEL_PATH,
        "Fake Audio": FAKE_AUDIO_PATH,
        "Real Audio": REAL_AUDIO_PATH
    }
    
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        
        if exists and os.path.isdir(path):
            try:
                files = os.listdir(path)
                print(f"    Contains {len(files)} items")
                if name in ["Fake Audio", "Real Audio"]:
                    audio_files = [f for f in files if f.endswith(('.wav', '.mp3', '.flac'))]
                    print(f"    Audio files: {len(audio_files)}")
                    if audio_files:
                        print(f"    Sample files: {audio_files[:3]}")
            except PermissionError:
                print(f"    Permission denied to list contents")
    
    print("=== END PATH DEBUG ===\n")

def debug_saved_preprocessor():
    """Debug the saved preprocessor with exact path"""
    print("=== DEBUGGING SAVED PREPROCESSOR ===")
    
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            data = joblib.load(PREPROCESSOR_PATH)
            print(f"✅ Preprocessor loaded successfully")
            print(f"Keys in saved data: {list(data.keys())}")
            
            if 'feature_columns' in data:
                print(f"Feature columns count: {len(data['feature_columns'])}")
                print(f"First 10 feature columns: {data['feature_columns'][:10]}")
                print(f"Last 10 feature columns: {data['feature_columns'][-10:]}")
                
                # Check if we have the expected 768 Wav2Vec2 features
                embed_features = [col for col in data['feature_columns'] if col.startswith('embed_')]
                print(f"Embed features found: {len(embed_features)}")
            
            if 'label_encoder' in data:
                print(f"Label encoder classes: {list(data['label_encoder'].classes_)}")
            
            if 'scaler' in data:
                print(f"Scaler type: {type(data['scaler'])}")
                if hasattr(data['scaler'], 'n_features_in_'):
                    print(f"Scaler expects {data['scaler'].n_features_in_} features")
                    
        except Exception as e:
            print(f"❌ Error loading preprocessor: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    else:
        print(f"❌ Preprocessor file not found: {PREPROCESSOR_PATH}")
    
    print("=== END PREPROCESSOR DEBUG ===\n")

def debug_saved_model():
    """Debug the saved model with exact path"""
    print("=== DEBUGGING SAVED MODEL ===")
    
    if os.path.exists(MODEL_PATH):
        try:
            model_data = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully")
            print(f"Model type: {type(model_data)}")
            
            # Check if it's a sklearn model with n_features_in_
            if hasattr(model_data, 'n_features_in_'):
                print(f"Model expects {model_data.n_features_in_} features")
            
            # If it's a custom class, check its attributes
            if hasattr(model_data, '__dict__'):
                attrs = [attr for attr in dir(model_data) if not attr.startswith('_')]
                print(f"Model attributes: {attrs[:10]}...")
                
                if hasattr(model_data, 'model'):
                    print(f"Inner model type: {type(model_data.model)}")
                    if hasattr(model_data.model, 'n_features_in_'):
                        print(f"Inner model expects {model_data.model.n_features_in_} features")
                        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
    
    print("=== END MODEL DEBUG ===\n")

def test_wav2vec2_extraction():
    """Test raw Wav2Vec2 feature extraction"""
    print("=== TESTING WAV2VEC2 EXTRACTION ===")
    
    # Find a sample audio file
    sample_file = None
    for audio_dir in [FAKE_AUDIO_PATH, REAL_AUDIO_PATH]:
        if os.path.exists(audio_dir):
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
            if audio_files:
                sample_file = os.path.join(audio_dir, audio_files[0])
                break
    
    if not sample_file:
        print("❌ No sample audio file found")
        return
    
    print(f"Testing with: {sample_file}")
    
    try:
        # Load Wav2Vec2
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        print("✅ Wav2Vec2 models loaded")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(sample_file)
        print(f"Audio loaded: shape={waveform.shape}, sr={sample_rate}")
        
        # Resample if needed
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            print("Audio resampled to 16kHz")
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print("Converted to mono")
        
        # Process with Wav2Vec2
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
        print(f"Wav2Vec2 input shape: {inputs.shape}")
        
        with torch.no_grad():
            outputs = model(inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            print(f"✅ Embedding extracted successfully!")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Expected: (768,) - Got: {embedding.shape}")
            
            if embedding.shape[0] == 768:
                print("🎉 Correct feature dimension!")
            else:
                print("⚠️ Unexpected feature dimension!")
                
    except Exception as e:
        print(f"❌ Error in Wav2Vec2 extraction: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    print("=== END WAV2VEC2 TEST ===\n")

def test_full_preprocessing_pipeline():
    """Test the complete preprocessing pipeline"""
    print("=== TESTING FULL PREPROCESSING PIPELINE ===")
    
    # Find a sample audio file
    sample_file = None
    for audio_dir in [FAKE_AUDIO_PATH, REAL_AUDIO_PATH]:
        if os.path.exists(audio_dir):
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
            if audio_files:
                sample_file = os.path.join(audio_dir, audio_files[0])
                break
    
    if not sample_file:
        print("❌ No sample audio file found")
        return
    
    print(f"Testing preprocessing with: {sample_file}")
    
    try:
        # Import your preprocessing class
        import sys
        sys.path.append(BASE_PATH)
        from src.preprocessing import AudioPreprocessor
        
        # Create and load preprocessor
        preprocessor = AudioPreprocessor()
        preprocessor.load_preprocessor(PREPROCESSOR_PATH)
        print("✅ Preprocessor loaded")
        
        # Test preprocessing
        result = preprocessor.preprocess_single_file(sample_file)
        print(f"✅ Preprocessing successful!")
        print(f"Result shape: {result.shape}")
        print(f"Expected: (1, 768) - Got: {result.shape}")
        
        if result.shape[1] == 768:
            print("🎉 Correct preprocessing output!")
        else:
            print(f"⚠️ Unexpected preprocessing output! Expected 768 features, got {result.shape[1]}")
            
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    print("=== END PREPROCESSING PIPELINE TEST ===\n")

def test_model_prediction():
    """Test model prediction with correct features"""
    print("=== TESTING MODEL PREDICTION ===")
    
    try:
        # Create dummy 768-feature input
        dummy_input = np.random.randn(1, 768).astype(np.float32)
        print(f"Created dummy input: {dummy_input.shape}")
        
        # Load model
        model_data = joblib.load(MODEL_PATH)
        print("✅ Model loaded")
        
        # Test prediction
        if hasattr(model_data, 'predict'):
            prediction = model_data.predict(dummy_input)
            print(f"✅ Model prediction successful: {prediction}")
        elif hasattr(model_data, 'model') and hasattr(model_data.model, 'predict'):
            prediction = model_data.model.predict(dummy_input)
            print(f"✅ Inner model prediction successful: {prediction}")
        else:
            print("❌ Cannot find predict method")
            
    except Exception as e:
        print(f"❌ Error in model prediction: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    print("=== END MODEL PREDICTION TEST ===\n")

def main():
    """Run all debug functions"""
    print("🔍 COMPREHENSIVE DEBUGGING SESSION")
    print("=" * 50)
    
    debug_paths()
    debug_saved_preprocessor()
    debug_saved_model()
    test_wav2vec2_extraction()
    test_full_preprocessing_pipeline()
    test_model_prediction()
    
    print("🔍 DEBUGGING SESSION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()