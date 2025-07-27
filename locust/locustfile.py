from locust import HttpUser, task, between
import os
import random
import tempfile
import wave
import numpy as np

class DeepfakeDetectionUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session"""
        self.test_files = self.create_test_audio_files()
    
    def create_test_audio_files(self):
        """Create synthetic audio files for testing"""
        test_files = []
        
        for i in range(3):  # Create 3 test files
            # Generate synthetic audio data
            duration = 2  # seconds
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Generate a simple sine wave with some noise
            frequency = random.randint(200, 800)
            audio_data = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            # Write WAV file
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            test_files.append(temp_file.name)
        
        return test_files
    
    @task(3)
    def predict_single_audio(self):
        """Test single audio prediction endpoint"""
        if not self.test_files:
            return
        
        file_path = random.choice(self.test_files)
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                response = self.client.post("/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('status') == 'success':
                        print(f"Prediction: {result.get('prediction')} (Confidence: {result.get('confidence'):.2f})")
                else:
                    print(f"Prediction failed with status {response.status_code}")
        
        except Exception as e:
            print(f"Error in predict_single_audio: {e}")
    
    @task(1)
    def predict_batch_audio(self):
        """Test batch audio prediction endpoint"""
        if len(self.test_files) < 2:
            return
        
        # Select 2 random files for batch prediction
        selected_files = random.sample(self.test_files, 2)
        
        try:
            files = []
            for i, file_path in enumerate(selected_files):
                with open(file_path, 'rb') as f:
                    files.append(('files', (f'test_audio_{i}.wav', f.read(), 'audio/wav')))
            
            response = self.client.post("/predict/batch", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Batch prediction completed for {len(result.get('results', []))} files")
            else:
                print(f"Batch prediction failed with status {response.status_code}")
        
        except Exception as e:
            print(f"Error in predict_batch_audio: {e}")
    
    @task(1)
    def health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"Health check: {health_data.get('status')}")
        else:
            print(f"Health check failed with status {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """Test model info endpoint"""
        response = self.client.get("/model/info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"Model type: {model_info.get('model_type')}")
        else:
            print(f"Model info failed with status {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """Test metrics endpoint"""
        response = self.client.get("/metrics")
        if response.status_code == 200:
            metrics = response.json()
            print(f"Metrics status: {metrics.get('status')}")
        else:
            print(f"Metrics failed with status {response.status_code}")
    
    @task(1)
    def get_retrain_status(self):
        """Test retrain status endpoint"""
        response = self.client.get("/retrain/status")
        if response.status_code == 200:
            status = response.json()
            total_pending = status.get('pending_files', {}).get('total', 0)
            print(f"Retrain status: {total_pending} pending files")
        else:
            print(f"Retrain status failed with status {response.status_code}")
    
    def on_stop(self):
        """Clean up test files"""
        for file_path in self.test_files:
            try:
                os.unlink(file_path)
            except:
                pass

class HighLoadUser(HttpUser):
    """High-load user for stress testing"""
    wait_time = between(0.1, 0.5)  # Faster requests
    
    def on_start(self):
        self.test_file = self.create_single_test_file()
    
    def create_single_test_file(self):
        """Create a single test file for high-load testing"""
        duration = 1  # Shorter duration for faster processing
        sample_rate = 16000  # Lower sample rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        with wave.open(temp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_file.name
    
    @task
    def rapid_predict(self):
        """Rapid prediction requests for load testing"""
        try:
            with open(self.test_file, 'rb') as f:
                files = {'file': ('load_test.wav', f, 'audio/wav')}
                self.client.post("/predict", files=files)
        except Exception as e:
            print(f"Error in rapid_predict: {e}")
    
    def on_stop(self):
        """Clean up test file"""
        try:
            os.unlink(self.test_file)
        except:
            pass
