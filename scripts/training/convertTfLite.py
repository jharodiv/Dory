import tensorflow as tf
import os

def convert_models():


    model_dir = "~/Documents/GitHub/Dory/features/models"
    model_paths = [
        '/Users/jemharoldcentino/Documents/GitHub/Dory/features/models/best_dory_model.h5',
        '/Users/jemharoldcentino/Documents/GitHub/Dory/features/models/dory_model_final.h5',
    ]
    
    for model_file in model_paths:
            if os.path.exists(model_file):
                print(f"\n🔄 Converting {os.path.basename(model_file)}...")
                print(f"   📁 Source: {model_file}")
                
                try:
                    # Load model
                    model = tf.keras.models.load_model(model_file)
                    
                    # Print model info
                    print(f"   Input shape: {model.input_shape}")
                    print(f"   Output shape: {model.output_shape}")
                    print(f"   Parameters: {model.count_params():,}")
                    
                    # Convert to TFLite
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_model = converter.convert()
                    
                    # Save TFLite version in same directory as source
                    model_dir = os.path.dirname(model_file)
                    model_name = os.path.splitext(os.path.basename(model_file))[0]
                    tflite_path = os.path.join(model_dir, f"{model_name}.tflite")
                    
                    with open(tflite_path, 'wb') as f:
                        f.write(tflite_model)
                    
                    # Check file sizes
                    h5_size = os.path.getsize(model_file) / (1024*1024)
                    tflite_size = os.path.getsize(tflite_path) / (1024*1024)
                    
                    print(f"   ✅ Converted: {tflite_path}")
                    print(f"   📏 Size: {h5_size:.2f}MB → {tflite_size:.2f}MB ({tflite_size/h5_size:.1f}x smaller)")
                    
                except Exception as e:
                    print(f"   ❌ Error converting {model_file}: {e}")
            else:
                print(f"❌ {model_file} not found")

convert_models()