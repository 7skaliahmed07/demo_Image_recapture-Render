import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import os

print("🔄 Converting Keras model to ONNX...")

# Load your model
model = tf.keras.models.load_model('final_model.keras')
print("✅ Model loaded")

# Convert to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=spec,
    output_path=output_path
)

print("✅ Model converted to ONNX!")

# Test the ONNX model
session = ort.InferenceSession(output_path)
print(f"📦 ONNX model inputs: {session.get_inputs()[0].name}")
print(f"📦 ONNX model outputs: {session.get_outputs()[0].name}")
print(f"📦 ONNX model size: {os.path.getsize('model.onnx') / (1024*1024):.1f} MB")