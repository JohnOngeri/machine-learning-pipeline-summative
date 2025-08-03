# 🎤 Deepfake Voice Detection System

A complete end-to-end machine learning system for detecting deepfake voices using audio feature analysis and classification algorithms.

## 📽️ YouTube Demo  
👉 [Click here to watch the video](https://youtu.be/tjtEcMNCHkM)


## 🎯 Project Overview

This project implements a comprehensive deepfake voice detection system that can:
- Classify audio clips as REAL or FAKE (deepfakes)
- Handle real-time predictions via REST API
- Support model retraining with user-uploaded data
- Provide interactive web interface for testing
- Handle high-traffic loads with performance monitoring
<img width="1362" height="633" alt="image" src="https://github.com/user-attachments/assets/ed263a1e-3a22-4398-8a6b-39b1559e311f" />

## 🏗️ Project Structure

```bash
machine learning pipeline summative/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .gitignore
│
├── notebook/
│   └── audio deep fake classification.ipynb       # Complete ML pipeline demonstration
│
├── src/
│   ├── preprocessing.py               # Audio feature extraction and preprocessing
│   ├── model.py                       # ML model training and evaluation
│   ├── prediction.py                  # Prediction service
│   └── retraining.py                  # Model retraining pipeline
│
├── data/
│   ├── train/                         # Training data directory
│   ├── test/                          # Test data directory
│   └── DATASET-balanced.csv          # Preprocessed features dataset
│
├── models/
│   ├── best_model.pkl                # Trained model
│   └── preprocessor.pkl              # Feature preprocessing pipeline
│
├── api/
│   └── main.py                       # FastAPI server for model serving
│
├── ui/
│   └── app.py                        # Streamlit dashboard
│
├── locust/
│   └── locustfile.py                 # Load testing script
│
└── scripts/
    ├── train_initial_model.py        # Initial model training script
    └── analyze_dataset.py
         set up project.py
         process raw auidon.py
          validate audio files.py
                      # Dataset analysis script
```
![WhatsApp Image 2025-08-01 at 11 44 17_3afea09b](https://github.com/user-attachments/assets/7c422ef0-15bd-4497-bb07-643891828cda)


## 🚀 Quick Start

### 1. Installation

\`\`\`bash
# Clone the repository
git clone <repository-url>
cd machine learning pipeline summative

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 2. Train Initial Model

\`\`\`bash
# Train the initial model with the provided dataset
python scripts/train_initial_model.py
\`\`\`

### 3. Start the API Server

\`\`\`bash
# Start FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000
\`\`\`

### 4. Launch the Web Interface

\`\`\`bash
# In a new terminal, start Streamlit dashboard
streamlit run ui/app.py --server.port 8501
\`\`\`

### 5. Access the Application

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

\`\`\`bash
# Build and start all services
docker-compose up --build

# Access services:
# - API: http://localhost:8000
# - Web UI: http://localhost:8501
# - Load Testing: http://localhost:8089
\`\`\`

### Individual Docker Containers

\`\`\`bash
# Build the image
docker build -t deepfake-detection .

# Run API server
docker run -p 8000:8000 -v $(pwd)/models:/app/models deepfake-detection

# Run Streamlit UI
docker run -p 8501:8501 -v $(pwd)/models:/app/models deepfake-detection streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
\`\`\`

## 📊 Features

### 🔍 Audio Analysis
- **Feature Extraction**: MFCC, Chroma, Spectral features, RMS, Zero-crossing rate
- **Preprocessing**: Standardization, missing value handling
- **Real-time Processing**: Single file and batch prediction support

### 🤖 Machine Learning
- **Multiple Algorithms**: MLP, Logistic Regression, CNN
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 🔄 Model Retraining
- **Dynamic Updates**: Add new training data via web interface
- **Automated Pipeline**: Preprocessing, training, and deployment
- **Backup System**: Automatic model versioning and rollback

### 🖥️ User Interface
- **File Upload**: Drag-and-drop audio file testing
- **Real-time Results**: Instant prediction with confidence scores
- **Visualizations**: Performance metrics, feature importance, prediction probabilities
- **Model Analytics**: Training history, cross-validation scores

### 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single file prediction |
| `/predict/batch` | POST | Batch file prediction |
| `/retrain/add-files` | POST | Add files for retraining |
| `/retrain` | POST | Start model retraining |
| `/retrain/status` | GET | Get retraining status |
| `/model/info` | GET | Get model information |
| `/metrics` | GET | Get performance metrics |


## 🧪 Load Testing

### Run Load Tests

\`\`\`bash
# Install locust if not already installed
pip install locust

# Run load test
locust -f locust/locustfile.py --host=http://localhost:8000

# Access load testing UI: http://localhost:8089
\`\`\`

### Performance Benchmarks
- **Concurrent Users**: 50+ users supported
- **Response Time**: <2 seconds for single predictions
- **Throughput**: 100+ requests/minute
- **Memory Usage**: <512MB under normal load

## 🔬 Dataset Information

### Source
- **Dataset**: Deep Voice Deepfake Voice Recognition
- **Source**: Kaggle Dataset – https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset/data?select=for-2sec

- **Size**: Balanced dataset with REAL and FAKE samples
-

## 🛠️ Development

### Running Tests

\`\`\`bash


### Code Quality

\`\`\`bash
# Format code
black src/ api/ ui/

# Lint code
flake8 src/ api/ ui/

# Type checking
mypy src/
\`\`\`

### Adding New Features

1. **New Model Algorithm**: Add to \`src/model.py\`
2. **New Features**: Extend \`src/preprocessing.py\`
3. **New API Endpoints**: Add to \`api/main.py\`
4. **UI Components**: Extend \`ui/app.py\`

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Found Error**
   \`\`\`bash
   # Train initial model
   python scripts/train_initial_model.py
   \`\`\`

2. **Audio File Processing Error**
   \`\`\`bash
   # Install audio processing dependencies
   sudo apt-get install libsndfile1 ffmpeg
   \`\`\`

3. **Memory Issues**
   \`\`\`bash
   # Reduce batch size or use smaller model
   # Check Docker memory limits
   \`\`\`

4. **Port Already in Use**
   \`\`\`bash
   # Change ports in docker-compose.yml or use different ports
   uvicorn api.main:app --port 8001
   \`\`\`

## 📝 Logging

Logs are stored in the \`logs/\` directory:
- \`api.log\`: API server logs
- \`retraining_log.txt\`: Model retraining history
- \`performance.log\`: Performance metrics

## 🔒 Security Considerations

- File upload validation and size limits
- Input sanitization for audio files
- Rate limiting on API endpoints
- Secure model file storage
- Environment variable protection

