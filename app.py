import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model
print("🚀 Loading TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded successfully!")

def predict(image):
    try:
        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # Format results - FIXED: No dropdown, clean output
        is_recaptured = prediction > 0.5
        label = "🖥️ Recaptured Image" if is_recaptured else "📷 Original Image"
        confidence = prediction if is_recaptured else 1 - prediction
        
        # Clean single line output without dropdown
        result_text = f"**{label}**\nConfidence: {confidence:.1%}"
        return result_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create interface - FIXED: Remove problematic components
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📁 Upload Image"),
    outputs=gr.Textbox(
        label="🎯 Detection Result",
        show_copy_button=True  # Better UX
    ),
    title="📸 Screen Recapture Detection",
    description="Upload an image to detect if it's original or recaptured from a screen",
    allow_flagging="never"  # Remove the flagging dropdown
)

demo.launch(show_api=False)  # Cleaner interface