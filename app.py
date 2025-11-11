from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gc

# AGGRESSIVE MEMORY OPTIMIZATION
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configure TensorFlow to use minimal memory
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = FastAPI(title="Screen Recapture Detection")

@app.on_event("startup")
async def startup_event():
    print("🚀 Loading TensorFlow Lite model...")
    try:
        # Use existing TFLite model
        app.state.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        app.state.interpreter.allocate_tensors()
        app.state.input_details = app.state.interpreter.get_input_details()
        app.state.output_details = app.state.interpreter.get_output_details()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")

@app.get("/")
async def home():
    return """
    <html>
        <head><title>Screen Recapture Detection</title>
        <style>
            body { font-family: Arial; max-width: 500px; margin: 50px auto; padding: 20px; }
            .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            .result { margin: 20px 0; padding: 15px; border-radius: 5px; }
            .original { background: #d4edda; color: #155724; }
            .recaptured { background: #f8d7da; color: #721c24; }
        </style>
        </head>
        <body>
            <h1>📸 Screen Recapture Detection</h1>
            <p>Upload an image to check if it's original or recaptured</p>
            
            <div class="upload-form">
                <form id="uploadForm">
                    <input type="file" id="fileInput" accept="image/*" required>
                    <br><br>
                    <button type="submit">Analyze Image</button>
                </form>
            </div>
            
            <div id="result"></div>
            
            <script>
                document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    const file = document.getElementById('fileInput').files[0];
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '<div class="result">Processing...</div>';
                    
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        
                        if (data.success) {
                            const label = data.prediction.label;
                            const confidence = (data.prediction.confidence * 100).toFixed(2);
                            resultDiv.innerHTML = `
                                <div class="result ${label.includes('Original') ? 'original' : 'recaptured'}">
                                    <h3>${label}</h3>
                                    <p>Confidence: ${confidence}%</p>
                                </div>`;
                        } else {
                            resultDiv.innerHTML = `<div class="result">Error: ${data.error}</div>`;
                        }
                    } catch (error) {
                        resultDiv.innerHTML = `<div class="result">Error: ${error.message}</div>`;
                    }
                });
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not hasattr(app.state, 'interpreter'):
        raise HTTPException(503, "Service starting...")
    
    try:
        # Read image with size limit
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(400, "Image too large (max 5MB)")
            
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((150, 150))  # Smaller size for less memory
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        interpreter = app.state.interpreter
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0]
        
        # Clean up memory immediately
        del img_array
        gc.collect()
        
        # Return result
        is_recaptured = prediction > 0.5
        label = "Recaptured Image" if is_recaptured else "Original Image"
        confidence = float(prediction if is_recaptured else 1 - prediction)
        
        return {
            "success": True,
            "prediction": {"label": label, "confidence": confidence}
        }
        
    except Exception as e:
        gc.collect()
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": hasattr(app.state, 'interpreter')}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)