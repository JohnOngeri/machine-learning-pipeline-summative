"""
Script to validate and inspect your audio files
"""

import os
import librosa
import pandas as pd
from loguru import logger

def validate_audio_files():
    """Validate audio files in your directories"""
    
    audio_base_path = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\data\AUDIO"
    
    results = []
    
    for label in ['REAL', 'FAKE']:
        label_dir = os.path.join(audio_base_path, label)
        
        if not os.path.exists(label_dir):
            logger.error(f"Directory not found: {label_dir}")
            continue
        
        logger.info(f"Validating {label} files...")
        
        files = [f for f in os.listdir(label_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for i, filename in enumerate(files[:10]):  # Check first 10 files
            file_path = os.path.join(label_dir, filename)
            
            try:
                # Load audio file
                y, sr = librosa.load(file_path, sr=None)
                duration = len(y) / sr
                
                results.append({
                    'filename': filename,
                    'label': label,
                    'duration': duration,
                    'sample_rate': sr,
                    'samples': len(y),
                    'status': 'OK'
                })
                
                if i < 3:  # Show details for first 3 files
                    logger.info(f"✅ {filename}: {duration:.2f}s, {sr}Hz, {len(y)} samples")
                
            except Exception as e:
                results.append({
                    'filename': filename,
                    'label': label,
                    'duration': None,
                    'sample_rate': None,
                    'samples': None,
                    'status': f'ERROR: {str(e)}'
                })
                logger.error(f"❌ {filename}: {str(e)}")
    
    # Create summary
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print("\n" + "="*50)
        print("AUDIO FILES VALIDATION SUMMARY")
        print("="*50)
        
        print(f"Total files checked: {len(df)}")
        print(f"Valid files: {len(df[df['status'] == 'OK'])}")
        print(f"Invalid files: {len(df[df['status'] != 'OK'])}")
        
        if len(df[df['status'] == 'OK']) > 0:
            valid_df = df[df['status'] == 'OK']
            print(f"\nValid files by label:")
            print(valid_df['label'].value_counts())
            
            print(f"\nDuration statistics:")
            print(f"Mean duration: {valid_df['duration'].mean():.2f}s")
            print(f"Min duration: {valid_df['duration'].min():.2f}s")
            print(f"Max duration: {valid_df['duration'].max():.2f}s")
            
            print(f"\nSample rate statistics:")
            print(valid_df['sample_rate'].value_counts())
        
        if len(df[df['status'] != 'OK']) > 0:
            print(f"\nProblematic files:")
            error_df = df[df['status'] != 'OK']
            for _, row in error_df.iterrows():
                print(f"- {row['filename']}: {row['status']}")
        
        print("="*50)
    else:
        logger.error("No audio files found to validate!")

if __name__ == "__main__":
    validate_audio_files()
