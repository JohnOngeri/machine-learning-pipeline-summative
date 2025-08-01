import joblib

# Load the individual artifacts
scaler = joblib.load('scaler_20250801_105126.pkl')
label_encoder = joblib.load('label_encoder_20250801_105126.pkl')

# Set the feature columns expected by your model
feature_columns = [f"f{i}" for i in range(768)]  # Wav2Vec2 output features

# Combine into a dictionary
preprocessor_data = {
    'scaler': scaler,
    'label_encoder': label_encoder,
    'feature_columns': feature_columns
}

# Save the combined preprocessor
joblib.dump(preprocessor_data, 'preprocessor.pkl')

print("✅ Combined preprocessor saved as preprocessor.pkl")
