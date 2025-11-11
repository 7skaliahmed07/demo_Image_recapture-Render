from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gc

# MEMORY OPTIMIZATION: Reduce TensorFlow memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

# Configure TensorFlow to use less memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Global variable for model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - load model
    global model
    print("🚀 Loading Keras model...")
    
    # MEMORY OPTIMIZATION: Clear any existing TensorFlow graphs
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Load model with optimizations
    model = tf.keras.models.load_model(
        'final_model.keras',
        compile=False  # MEMORY OPTIMIZATION: Don't compile (we're only doing inference)
    )
    print("✅ Model loaded successfully!")
    
    yield
    
    # Shutdown - cleanup
    print("👋 Shutting down...")
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()

app = FastAPI(title="Screen Recapture Detection", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Screen Recapture Detection</title>
            <style>
                body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
                .upload-form { border: 2px dashed #ccc; padding: 30px; text-align: center; margin: 20px 0; }
                button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
                .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
                .original { background: #d4edda; color: #155724; }
                .recaptured { background: #f8d7da; color: #721c24; }
                .warning { background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="warning">
                <strong>Note:</strong> This is a memory-optimized version. If you get errors, the service may be restarting due to memory limits.
            </div>
            
            <h1>📸 Screen Recapture Detection</h1>
            <p>Upload an image to detect if it's original or recaptured from a screen</p>
            
            <div class="upload-form">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="fileInput" name="file" accept="image/*" required>
                    <br><br>
                    <button type="submit">Analyze Image</button>
                </form>
            </div>
            
            <div id="result"></div>
            
            <script>
                document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    const fileInput = document.getElementById('fileInput');
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '<div class="result">Processing... Please wait.</div>';
                    
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        
                        if (data.success) {
                            const label = data.prediction.label;
                            const confidence = (data.prediction.confidence * 100).toFixed(2);
                            
                            resultDiv.innerHTML = `
                                <div class="result ${label.includes('Original') ? 'original' : 'recaptured'}">
                                    <h3>${label}</h3>
                                    <p>Confidence: ${confidence}%</p>
                                    <p>Raw score: ${data.prediction.raw_score.toFixed(4)}</p>
                                </div>
                            `;
                        } else {
                            resultDiv.innerHTML = `<div class="result" style="background: #fff3cd; color: #856404;">
                                <p>Error: ${data.error}</p>
                            </div>`;
                        }
                    } catch (error) {
                        resultDiv.innerHTML = `
                            <div class="result" style="background: #f8d7da; color: #721c24;">
                                <p>Error: ${error.message}</p>
                                <p>This might be due to memory limits. Try again in a moment.</p>
                            </div>`;
                    }
                });
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service may be restarting.")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    try:
        # MEMORY OPTIMIZATION: Process smaller images
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Resize to smaller dimensions to save memory
        img = img.resize((150, 150))  # Reduced from 224x224
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # MEMORY OPTIMIZATION: Clean up
        del img_array
        gc.collect()
        
        # Interpret results
        is_recaptured = prediction > 0.5
        label = "Recaptured Image" if is_recaptured else "Original Image"
        confidence = float(prediction if is_recaptured else 1 - prediction)
        
        return {
            "success": True,
            "prediction": {
                "label": label,
                "confidence": confidence,
                "raw_score": float(prediction)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health():
    global model
    return {
        "status": "healthy" if model is not None else "restarting",
        "model_loaded": model is not None,
        "message": "Screen Recapture Detection API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)  # Single worker to save memory