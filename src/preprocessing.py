import pandas as pd
import numpy as np
import librosa
import os
from typing import Tuple, List, Dict, Any
from loguru import logger
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class AudioPreprocessor:
    """Audio preprocessing pipeline for deepfake detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def extract_features(self, audio_path: str, sr: int = 22050) -> Dict[str, float]:
        """Extract audio features from a single audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=sr)
            
            # Extract features
            features = {}
            
            # Chroma STFT
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_stft'] = np.mean(chroma_stft)
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)
            features['rms'] = np.mean(rms)
            
            # Spectral Centroid
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = np.mean(spec_cent)
            
            # Spectral Bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth'] = np.mean(spec_bw)
            
            # Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['rolloff'] = np.mean(rolloff)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate'] = np.mean(zcr)
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            for i in range(20):
                features[f'mfcc{i+1}'] = np.mean(mfcc[i])
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def process_audio_directory(self, audio_dir: str) -> pd.DataFrame:
        """Process all audio files in a directory structure"""
        data = []
        
        for label in ['REAL', 'FAKE']:
            label_dir = os.path.join(audio_dir, label)
            if not os.path.exists(label_dir):
                logger.warning(f"Directory {label_dir} not found")
                continue
                
            logger.info(f"Processing {label} audio files...")
            
            for filename in os.listdir(label_dir):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(label_dir, filename)
                    features = self.extract_features(audio_path)
                    
                    if features:
                        features['LABEL'] = label
                        features['filename'] = filename
                        data.append(features)
        
        df = pd.DataFrame(data)
        logger.info(f"Processed {len(df)} audio files")
        return df
    
    def load_and_prepare_data(self, csv_path: str = None, audio_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for training"""
        if csv_path and os.path.exists(csv_path):
            logger.info(f"Loading data from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
        elif audio_dir and os.path.exists(audio_dir):
            logger.info(f"Processing audio files from: {audio_dir}")
            df = self.process_audio_directory(audio_dir)
            # Save processed data
            df.to_csv('data/DATASET-balanced.csv', index=False)
        else:
            raise ValueError("Either csv_path or audio_dir must be provided and exist")
        
        # Prepare features and labels
        feature_columns = [col for col in df.columns if col not in ['LABEL', 'filename']]
        self.feature_columns = feature_columns
        
        X = df[feature_columns].values
        y = df['LABEL'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Data shape: {X_scaled.shape}, Labels: {np.unique(y)}")
        return X_scaled, y_encoded
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def save_preprocessor(self, path: str = 'models/preprocessor.pkl'):
        """Save the preprocessor components"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        joblib.dump(preprocessor_data, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str = 'models/preprocessor.pkl'):
        """Load the preprocessor components"""
        preprocessor_data = joblib.load(path)
        self.scaler = preprocessor_data['scaler']
        self.label_encoder = preprocessor_data['label_encoder']
        self.feature_columns = preprocessor_data['feature_columns']
        logger.info(f"Preprocessor loaded from {path}")
    
    def preprocess_single_file(self, audio_path: str) -> np.ndarray:
        """Preprocess a single audio file for prediction"""
        features = self.extract_features(audio_path)
        if not features:
            raise ValueError(f"Could not extract features from {audio_path}")
        
        # Convert to DataFrame to ensure correct column order
        feature_df = pd.DataFrame([features])
        
        # Select only the feature columns used during training
        if self.feature_columns:
            feature_df = feature_df[self.feature_columns]
        
        # Handle missing columns
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        X = feature_df.values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
