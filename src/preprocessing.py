import os 
import sys
import joblib
import torch
import torchaudio
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Add project root to path
sys.path.insert(0, '..')

# Your exact paths
BASE_PATH = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative"
DEFAULT_PREPROCESSOR_PATH = os.path.join(BASE_PATH, "models", "preprocessor.pkl")
DEFAULT_DATA_PATH = os.path.join(BASE_PATH, "data")
AUDIO_DATA_PATH = os.path.join(BASE_PATH, "data", "AUDIO")

class AudioPreprocessor:
    """Handles Wav2Vec2-based audio feature extraction and preprocessing pipeline."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None

        # Load pre-trained Wav2Vec2 processor and model
        logger.info("Loading Wav2Vec2 models...")
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        logger.info("✅ Wav2Vec2 models loaded successfully")

    def extract_features(self, audio_path: str, sr: int = 16000) -> Dict[str, float] | None:
        """Extract Wav2Vec2 embedding features from a single audio file."""
        try:
            logger.debug(f"🎵 Loading audio file: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None
            
            waveform, sample_rate = torchaudio.load(audio_path)
            logger.debug(f"📊 Original audio shape: {waveform.shape}, sample_rate: {sample_rate}")

            # Resample if needed
            if sample_rate != sr:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)(waveform)
                logger.debug(f"🔄 Resampled to {sr}Hz")

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                logger.debug("🎧 Converted to mono")

            logger.debug(f"📏 Final waveform shape: {waveform.shape}")

            # Process with Wav2Vec2
            inputs = self.wav2vec2_processor(
                waveform.squeeze().numpy(), 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_values
            
            logger.debug(f"🤖 Wav2Vec2 input shape: {inputs.shape}")

            with torch.no_grad():
                outputs = self.wav2vec2_model(inputs)
                # Get the last hidden state and average across time dimension
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                logger.debug(f"🧠 Raw Wav2Vec2 embedding shape: {embedding.shape}")
                logger.debug(f"📋 Embedding type: {type(embedding)}")
                
                # Ensure we have the expected 768 features
                if len(embedding.shape) == 0:  # scalar
                    logger.error("Embedding is a scalar, expected array")
                    return None
                
                if embedding.shape[0] != 768:
                    logger.error(f"❌ Expected 768 features, got {embedding.shape[0]}")
                    logger.error(f"Full embedding shape: {embedding.shape}")
                    return None
                
                logger.debug(f"✅ Correct embedding size: {embedding.shape[0]} features")

            # Convert to feature dictionary
            features = {f"embed_{i}": float(val) for i, val in enumerate(embedding)}
            logger.debug(f"📝 Created {len(features)} feature values")
            
            # Verify all values are finite
            finite_check = all(np.isfinite(val) for val in features.values())
            if not finite_check:
                logger.warning("⚠️ Some feature values are not finite (inf/nan)")
            
            return features

        except Exception as e:
            logger.error(f"💥 Failed to process {audio_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def process_audio_directory(self, audio_dir: str = None) -> pd.DataFrame:
        """Processes all labeled audio files inside a given directory structure."""
        if audio_dir is None:
            audio_dir = AUDIO_DATA_PATH
            
        logger.info(f"🎵 Processing audio directory: {audio_dir}")
        
        data = []
        for label in ['real', 'fake']:
            label_dir = os.path.join(audio_dir, label)
            if not os.path.exists(label_dir):
                logger.warning(f"⚠️ Missing folder: {label_dir}")
                continue

            logger.info(f"🔍 Extracting features from {label.upper()} audio in: {label_dir}")
            
            audio_files = [f for f in os.listdir(label_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
            logger.info(f"📁 Found {len(audio_files)} audio files in {label} folder")

            for i, file in enumerate(audio_files):
                path = os.path.join(label_dir, file)
                logger.debug(f"Processing {i+1}/{len(audio_files)}: {file}")
                
                features = self.extract_features(path)
                if features:
                    features['LABEL'] = label
                    features['filename'] = file
                    data.append(features)
                    logger.debug(f"✅ Successfully processed {file}")
                else:
                    logger.warning(f"❌ Failed to process {file}")

        df = pd.DataFrame(data)
        logger.info(f"📊 Total files processed successfully: {len(df)}")
        
        # Log feature information
        if not df.empty:
            feature_cols = [col for col in df.columns if col not in ['LABEL', 'filename']]
            logger.info(f"🧮 Features extracted per file: {len(feature_cols)}")
            logger.info(f"📋 Labels distribution: {df['LABEL'].value_counts().to_dict()}")
        
        return df

    def load_and_prepare_data(self, csv_path: str = None, audio_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Loads data from CSV or processes audio files, returning scaled features and encoded labels."""
        if csv_path and os.path.exists(csv_path):
            logger.info(f"📄 Reading data from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
        elif audio_dir and os.path.exists(audio_dir):
            logger.info(f"🎵 Processing audio from: {audio_dir}")
            df = self.process_audio_directory(audio_dir)
            
            # Save processed data
            os.makedirs(DEFAULT_DATA_PATH, exist_ok=True)
            csv_save_path = os.path.join(DEFAULT_DATA_PATH, 'DATASET-balanced.csv')
            df.to_csv(csv_save_path, index=False)
            logger.info(f"💾 Saved processed data to: {csv_save_path}")
        else:
            # Try default audio directory
            default_audio = AUDIO_DATA_PATH
            if os.path.exists(default_audio):
                logger.info(f"🎵 Using default audio directory: {default_audio}")
                df = self.process_audio_directory(default_audio)
                
                os.makedirs(DEFAULT_DATA_PATH, exist_ok=True)
                csv_save_path = os.path.join(DEFAULT_DATA_PATH, 'DATASET-balanced.csv')
                df.to_csv(csv_save_path, index=False)
                logger.info(f"💾 Saved processed data to: {csv_save_path}")
            else:
                raise ValueError(f"No valid data source found. Checked: csv_path={csv_path}, audio_dir={audio_dir}, default={default_audio}")

        if df.empty:
            raise ValueError("No data was loaded or processed")

        self.feature_columns = [col for col in df.columns if col not in ['LABEL', 'filename']]
        logger.info(f"🔢 Feature columns identified: {len(self.feature_columns)}")
        
        # Verify we have 768 features
        if len(self.feature_columns) != 768:
            logger.warning(f"⚠️ Expected 768 features, got {len(self.feature_columns)}")
        
        logger.debug(f"First 10 feature columns: {self.feature_columns[:10]}")
        logger.debug(f"Last 10 feature columns: {self.feature_columns[-10:]}")
        
        X = df[self.feature_columns].values
        y = df['LABEL'].values

        logger.info(f"📊 Raw data shape: X={X.shape}, y={y.shape}")
        
        # Handle NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            logger.warning(f"⚠️ Found {nan_count} NaN values, replacing with 0")
        X = np.nan_to_num(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f"🏷️ Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"📏 Final scaled data shape: X={X_scaled.shape}, y={y_encoded.shape}")
        logger.info(f"🎯 Classes: {list(self.label_encoder.classes_)}")
        
        return X_scaled, y_encoded

    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Splits dataset into training and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def save_preprocessor(self, path: str = None):
        """Saves scaler, encoder, and feature column metadata."""
        if path is None:
            path = DEFAULT_PREPROCESSOR_PATH
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(preprocessor_data, path)
        logger.info(f"💾 Preprocessor saved to: {path}")
        logger.info(f"📊 Saved {len(self.feature_columns)} feature columns")
        
        # Debug: Verify what was saved
        try:
            test_load = joblib.load(path)
            logger.debug(f"✅ Verification: Successfully loaded {len(test_load['feature_columns'])} feature columns")
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")

    def load_preprocessor(self, path: str = None):
        """Loads pre-saved preprocessing components."""
        if path is None:
            path = DEFAULT_PREPROCESSOR_PATH
            
        logger.info(f"📂 Loading preprocessor from: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        
        preprocessor = joblib.load(path)
        self.scaler = preprocessor['scaler']
        self.label_encoder = preprocessor['label_encoder']
        self.feature_columns = preprocessor['feature_columns']
        
        logger.info(f"✅ Preprocessor loaded successfully")
        logger.info(f"🔢 Feature columns loaded: {len(self.feature_columns)}")
        
        # Verify expected feature count
        if len(self.feature_columns) != 768:
            logger.warning(f"⚠️ Expected 768 features, loaded {len(self.feature_columns)}")
        else:
            logger.info(f"✅ Correct feature count: {len(self.feature_columns)}")
            
        logger.debug(f"First 10 features: {self.feature_columns[:10]}")
        logger.debug(f"Last 10 features: {self.feature_columns[-10:]}")
        logger.info(f"🏷️ Label classes: {list(self.label_encoder.classes_)}")

    def preprocess_single_file(self, audio_path: str) -> np.ndarray:
        """Prepares a single file for inference using the trained pipeline."""
        logger.debug(f"🎯 Starting preprocessing for: {audio_path}")
        
        # Check if preprocessor is loaded
        if self.feature_columns is None:
            raise ValueError("❌ Preprocessor not loaded. Call load_preprocessor() first.")
        
        logger.debug(f"📊 Expected feature columns: {len(self.feature_columns)}")
        
        # Extract features using Wav2Vec2
        features = self.extract_features(audio_path)
        if features is None:
            raise ValueError(f"❌ Feature extraction failed for: {audio_path}")

        logger.debug(f"🧠 Raw features extracted: {len(features)}")
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        logger.debug(f"📋 DataFrame shape after feature extraction: {df.shape}")
        logger.debug(f"📊 DataFrame columns: {len(df.columns)}")
        
        # Ensure all expected columns exist
        missing_cols = []
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
                missing_cols.append(col)
        
        if missing_cols:
            logger.warning(f"⚠️ Added {len(missing_cols)} missing columns with zeros")
            logger.debug(f"Missing columns: {missing_cols[:10]}...")
        
        # Select only the feature columns in the correct order
        df = df[self.feature_columns]
        logger.debug(f"📏 Final DataFrame shape: {df.shape}")
        
        # Convert to numpy and handle NaN values
        X = np.nan_to_num(df.values)
        logger.debug(f"🔢 Numpy array shape before scaling: {X.shape}")
        
        # Verify shape
        if X.shape[1] != len(self.feature_columns):
            raise ValueError(f"❌ Shape mismatch: expected {len(self.feature_columns)} features, got {X.shape[1]}")
        
        # Apply scaling
        X_scaled = self.scaler.transform(X)
        logger.debug(f"📏 Final scaled array shape: {X_scaled.shape}")
        
        # Final verification
        if X_scaled.shape[1] != 768:
            logger.error(f"❌ CRITICAL: Final output has {X_scaled.shape[1]} features, expected 768!")
            raise ValueError(f"Feature dimension error: got {X_scaled.shape[1]}, expected 768")
        
        logger.debug(f"✅ Preprocessing successful: {X_scaled.shape}")
        return X_scaled


# Standalone debugging functions
def debug_with_your_paths():
    """Debug function using your exact paths"""
    print("🔍 DEBUGGING WITH YOUR EXACT PATHS")
    print("=" * 50)
    
    # Test paths
    paths = {
        "Base": BASE_PATH,
        "Preprocessor": DEFAULT_PREPROCESSOR_PATH,
        "Audio Data": AUDIO_DATA_PATH,
        "Fake Audio": os.path.join(AUDIO_DATA_PATH, "fake"),
        "Real Audio": os.path.join(AUDIO_DATA_PATH, "real")
    }
    
    for name, path in paths.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"{exists} {name}: {path}")
    
    # Test feature extraction
    fake_dir = os.path.join(AUDIO_DATA_PATH, "fake")
    if os.path.exists(fake_dir):
        audio_files = [f for f in os.listdir(fake_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        if audio_files:
            sample_file = os.path.join(fake_dir, audio_files[0])
            print(f"\n🎵 Testing with sample file: {sample_file}")
            
            preprocessor = AudioPreprocessor()
            features = preprocessor.extract_features(sample_file)
            
            if features:
                print(f"✅ Feature extraction successful: {len(features)} features")
            else:
                print("❌ Feature extraction failed")
    
    print("=" * 50)

if __name__ == "__main__":
    debug_with_your_paths()