from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger
import uvicorn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import sys
sys.path.insert(0, '..')

from src.prediction import DeepfakePredictionService
from src.retraining import RetrainingService

# Initialize FastAPI appssss
app = FastAPI(
    title="Deepfake Voice Detection API",
    description="API for detecting deepfake voices using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
prediction_service = DeepfakePredictionService()
retraining_service = RetrainingService()

# Pydantic models
class PredictionResponse(BaseModel):
    filename: str
    prediction: Optional[str]
    confidence: Optional[float]
    probabilities: Optional[dict]
    status: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool

class RetrainRequest(BaseModel):
    include_original_data: bool = True
    model_type: str = "random_forest"

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Deepfake Voice Detection API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    health_status = prediction_service.health_check()
    return HealthResponse(**health_status)

@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """Predict if an audio file is real or fake"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only .wav, .mp3, and .flac files are supported."
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Make prediction
        result = prediction_service.predict_single_file(tmp_path)
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict multiple audio files"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    temp_files = []
    
    try:
        # Save all files temporarily
        for file in files:
            if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
                results.append({
                    'filename': file.filename,
                    'prediction': None,
                    'confidence': None,
                    'probabilities': None,
                    'status': 'error',
                    'error': 'Invalid file type'
                })
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_files.append(tmp_file.name)
        
        # Make predictions
        if temp_files:
            batch_results = prediction_service.predict_batch(temp_files)
            results.extend(batch_results)
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
        
    finally:
        # Clean up temporary files
        for tmp_path in temp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

@app.post("/retrain/add-files")
async def add_retrain_files(
    files: List[UploadFile] = File(...),
    labels: List[str] = Form(...)
):
    """Add files for retraining"""
    
    if len(files) != len(labels):
        raise HTTPException(status_code=400, detail="Number of files must match number of labels")
    
    if len(files) > 20:  # Limit number of files
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed")
    
    temp_files = []
    file_info = []
    
    try:
        # Save files temporarily and prepare file info
        for file, label in zip(files, labels):
            if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
                continue
                
            if label.upper() not in ['REAL', 'FAKE']:
                continue
            
            # Save to retrain directory
            label_dir = os.path.join('data', 'retrain', label.upper())
            os.makedirs(label_dir, exist_ok=True)
            
            file_path = os.path.join(label_dir, file.filename)
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            
            file_info.append({
                'path': file_path,
                'label': label.upper()
            })
        
        # Add files to retraining service
        result = retraining_service.add_training_files(file_info)
        
        return {
            "message": "Files added for retraining",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error adding retrain files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add files: {str(e)}")

@app.post("/retrain")
async def retrain_model(request: RetrainRequest):
    """Retrain the model with new data"""
    
    try:
        # Check if there are files to retrain with
        status = retraining_service.get_retrain_status()
        if not status['ready_for_retraining']:
            raise HTTPException(status_code=400, detail="No files available for retraining")
        
        # Start retraining
        result = retraining_service.retrain_model(
            include_original_data=request.include_original_data,
            model_type=request.model_type
        )
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        # Reload the prediction service with new model
        prediction_service.load_components()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/retrain/status")
async def get_retrain_status():
    """Get current retraining status"""
    return retraining_service.get_retrain_status()

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    return prediction_service.get_model_info()

@app.get("/metrics")
async def get_metrics():
    """Get current model performance metrics"""
    model_info = prediction_service.get_model_info()
    
    if 'training_history' in model_info:
        return {
            "model_metrics": model_info['training_history'],
            "model_type": model_info.get('model_type'),
            "status": "available"
        }
    else:
        return {"status": "metrics_not_available"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
