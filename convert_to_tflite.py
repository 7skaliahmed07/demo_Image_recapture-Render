import tensorflow as tf
import os
import numpy as np

def convert_keras_to_tflite():
    """
    Convert Keras model to TensorFlow Lite format
    """
    print("🔄 Starting Keras to TensorFlow Lite conversion...")
    
    try:
        # Load your Keras model
        print("📥 Loading Keras model...")
        model = tf.keras.models.load_model('final_model.keras')
        
        # Display model info
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        print(f"📊 Model layers: {len(model.layers)}")
        
        # Convert to TensorFlow Lite
        print("🛠️ Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimization settings for better performance
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        
        # Optional: For even smaller size (may reduce accuracy slightly)
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save the converted model
        output_path = "model.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Display conversion results
        original_size = os.path.getsize('final_model.keras') / (1024*1024)
        tflite_size = os.path.getsize(output_path) / (1024*1024)
        reduction = ((original_size - tflite_size) / original_size) * 100
        
        print("🎉 Conversion completed successfully!")
        print(f"📦 Original Keras model: {original_size:.1f} MB")
        print(f"📦 TensorFlow Lite model: {tflite_size:.1f} MB")
        print(f"📉 Size reduction: {reduction:.1f}%")
        
        # Test the converted model
        print("🧪 Testing converted model...")
        test_converted_model(output_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def test_converted_model(model_path):
    """
    Test that the converted model works correctly
    """
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ TFLite model test passed!")
        print(f"📥 Input details: {input_details[0]['shape']} ({input_details[0]['dtype']})")
        print(f"📤 Output details: {output_details[0]['shape']} ({output_details[0]['dtype']})")
        
        # Test with dummy data
        input_shape = input_details[0]['shape']
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"🧪 Test prediction shape: {output_data.shape}")
        print("🎯 Model is ready for deployment!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

if __name__ == "__main__":
    # Check if model file exists
    if not os.path.exists('final_model.keras'):
        print("❌ Error: 'final_model.keras' file not found!")
        print("💡 Make sure your Keras model is in the same directory")
    else:
        convert_keras_to_tflite()