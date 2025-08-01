import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import time
from datetime import datetime
import json
import sys
import json
sys.path.insert(0, '..')
# Page configuration
st.set_page_config(
    page_title="Deepfake Voice Detection",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4ECDC4;
    }
    .prediction-real {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .prediction-fake {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_audio(file):
    """Send audio file for prediction"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        return response.json()
    except:
        return {}

def get_metrics():
    """Get model metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        return response.json()
    except:
        return {}

def retrain_model(include_original=True, model_type="random_forest"):
    """Trigger model retraining"""
    try:
        data = {
            "include_original_data": include_original,
            "model_type": model_type
        }
        response = requests.post(f"{API_BASE_URL}/retrain", json=data, timeout=300)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def add_retrain_files(files, labels):
    """Add files for retraining"""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file, file.type)))
        
        form_data = {"labels": labels}
        
        response = requests.post(
            f"{API_BASE_URL}/retrain/add-files",
            files=files_data,
            data=form_data,
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    # Header
    st.markdown('<h1 class="main-header">🎤 Deepfake Voice Detection System</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not running. Please start the API server first.")
        st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🔍 Prediction", "📊 Model Analytics", "🔄 Retraining", "ℹ️ System Info"]
    )
    
    if page == "🔍 Prediction":
        prediction_page()
    elif page == "📊 Model Analytics":
        analytics_page()
    elif page == "🔄 Retraining":
        retraining_page()
    elif page == "ℹ️ System Info":
        system_info_page()

def prediction_page():
    st.header("Audio Prediction")
    st.write("Upload an audio file to check if it's real or deepfake generated.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac'],
        help="Supported formats: WAV, MP3, FLAC"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        # Audio player
        st.audio(uploaded_file)
        
        # Predict button
        if st.button("🔍 Analyze Audio", type="primary"):
            with st.spinner("Analyzing audio... This may take a few seconds."):
                result = predict_audio(uploaded_file)
            
            if result.get("status") == "success":
                prediction = result["prediction"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]
                
                # Display result
                if prediction == "REAL":
                    st.markdown(f"""
                    <div class="prediction-real">
                        <h3>✅ REAL VOICE DETECTED</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-fake">
                        <h3>⚠️ DEEPFAKE DETECTED</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability chart
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['REAL', 'FAKE'],
                            y=[probabilities['REAL'], probabilities['FAKE']],
                            marker_color=['#28a745', '#dc3545']
                        )
                    ])
                    fig.update_layout(
                        title="Prediction Probabilities",
                        yaxis_title="Probability",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence Level"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#4ECDC4"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")



def analytics_page():
    st.header("Model Analytics")

    # Get model metrics
    metrics = get_metrics()
    model_info = get_model_info()

    if metrics.get("status") == "available":
        training_history = metrics["model_metrics"]

        # ✅ FIX: parse the string into a dictionary if needed
        if isinstance(training_history, str):
            try:
                training_history = json.loads(training_history)
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse training history: {e}")
                return
            st.write("Raw training history:", training_history)
        # ✅ Choose model: CNN, Logistic Regression, or MLP
        selected_model = "Logistic Regression"  # Change this dynamically if needed
        training_history = training_history.get(selected_model, {})

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            f1 = training_history.get("f1_score", 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>CV F1 Score</h4>
                <h2>{f1:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

        
        with col2:
            accuracy = training_history.get("accuracy", 0)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Validation Accuracy</h4>
                <h2>{accuracy:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            precision = training_history.get("precision", 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Precision</h4>
                <h2>{precision:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            recall = training_history.get("recall", 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Recall</h4>
                <h2>{recall:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Cross-validation scores
        if "cv_scores" in training_history:
            st.subheader("Cross-Validation Scores")
            cv_scores = training_history["cv_scores"]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cv_scores) + 1)),
                y=cv_scores,
                mode='lines+markers',
                name='CV F1 Score',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8)
            ))
            fig.add_hline(y=np.mean(cv_scores), line_dash="dash", 
                         annotation_text=f"Mean: {np.mean(cv_scores):.3f}")
            fig.update_layout(
                title="Cross-Validation F1 Scores",
                xaxis_title="Fold",
                yaxis_title="F1 Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison metrics
        if training_history:
            st.subheader("Model Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Score': [
                    accuracy,
                    precision,
                    recall,
                    f1
                ]
            })
            
            fig = px.bar(
                metrics_df, 
                x='Metric', 
                y='Score',
                color='Score',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(title="Model Performance Overview")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Model metrics not available. Train a model first.")

def retraining_page():
    st.header("Model Retraining")
    st.write("Upload new audio files to improve the model's performance.")
    
    # Get retrain status
    try:
        response = requests.get(f"{API_BASE_URL}/retrain/status")
        retrain_status = response.json()
    except:
        retrain_status = {"pending_files": {"REAL": 0, "FAKE": 0, "total": 0}}
    
    # Display current status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pending REAL files", retrain_status["pending_files"]["REAL"])
    with col2:
        st.metric("Pending FAKE files", retrain_status["pending_files"]["FAKE"])
    with col3:
        st.metric("Total pending", retrain_status["pending_files"]["total"])
    
    st.divider()
    
    # File upload for retraining
    st.subheader("Add Training Files")
    
    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=['wav', 'mp3', 'flac'],
        accept_multiple_files=True,
        help="Upload multiple audio files for retraining"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files:")
        
        # Label assignment
        labels = []
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📁 {file.name}")
            with col2:
                label = st.selectbox(
                    "Label",
                    ["REAL", "FAKE"],
                    key=f"label_{i}",
                    label_visibility="collapsed"
                )
                labels.append(label)
        
        # Add files button
        if st.button("📤 Add Files for Retraining", type="primary"):
            with st.spinner("Adding files..."):
                result = add_retrain_files(uploaded_files, labels)
            
            if "error" not in result:
                st.success("✅ Files added successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
    
    st.divider()
    
    # Retraining options
    st.subheader("Retraining Options")
    
    col1, col2 = st.columns(2)
    with col1:
        include_original = st.checkbox(
            "Include original training data",
            value=True,
            help="Whether to include the original dataset in retraining"
        )
    
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["random_forest", "logistic_regression", "svm"],
            help="Choose the model algorithm for retraining"
        )
    
    # Retrain button
    if retrain_status["pending_files"]["total"] > 0:
        if st.button("🔄 Start Retraining", type="primary"):
            with st.spinner("Retraining model... This may take several minutes."):
                result = retrain_model(include_original, model_type)
            
            if result.get("status") == "success":
                st.success("✅ Model retrained successfully!")
                
                # Display results
                st.subheader("Retraining Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("New Samples", result["new_samples_count"])
                    st.metric("Total Samples", result["total_samples_count"])
                
                with col2:
                    eval_results = result["evaluation_results"]
                    st.metric("Test Accuracy", f"{eval_results['metrics']['accuracy']:.3f}")
                    st.metric("Test F1 Score", f"{eval_results['metrics']['f1']:.3f}")
                
                st.rerun()
            else:
                st.error(f"❌ Retraining failed: {result.get('message', 'Unknown error')}")
    else:
        st.info("ℹ️ No files available for retraining. Please add some files first.")

def system_info_page():
    st.header("System Information")
    
    # API health
    api_healthy = check_api_health()
    status_color = "🟢" if api_healthy else "🔴"
    st.write(f"**API Status:** {status_color} {'Healthy' if api_healthy else 'Unhealthy'}")
    
    # Model info
    model_info = get_model_info()
    
    if model_info:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
            st.write(f"**Model Loaded:** {'✅' if model_info.get('model_loaded') else '❌'}")
            st.write(f"**Preprocessor Loaded:** {'✅' if model_info.get('preprocessor_loaded') else '❌'}")
        
        with col2:
            if model_info.get('best_params'):
                st.write("**Best Parameters:**")
                for param, value in model_info['best_params'].items():
                    st.write(f"- {param}: {value}")
    
   

    # System metrics
    st.subheader("System Metrics")

    # Full absolute paths
    data_dir = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\data"
    models_dir = r"C:\Users\HP\OneDrive\Desktop\machine learning pipeline summative\models"

    try:
        # Check Data Directory
        if os.path.exists(data_dir):
            st.write(f"**Data Directory:** ✅ {data_dir}")
        else:
            st.write(f"**Data Directory:** ❌ {data_dir} (not found)")

        # Check Models Directory
        if os.path.exists(models_dir):
            st.write(f"**Models Directory:** ✅ {models_dir}")
            model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.h5', '.tf'))]
            st.write(f"**Model Files:** {len(model_files)} found")
        else:
            st.write(f"**Models Directory:** ❌ {models_dir} (not found)")

    except Exception as e:
        st.write(f"**File System Error:** {str(e)}")

    
    # API endpoints
    st.subheader("Available API Endpoints")
    endpoints = [
        "GET /health - Health check",
        "POST /predict - Single file prediction",
        "POST /predict/batch - Batch prediction",
        "POST /retrain/add-files - Add files for retraining",
        "POST /retrain - Start retraining",
        "GET /retrain/status - Get retraining status",
        "GET /model/info - Get model information",
        "GET /metrics - Get model metrics"
    ]
    
    for endpoint in endpoints:
        st.write(f"- {endpoint}")

if __name__ == "__main__":
    main()
