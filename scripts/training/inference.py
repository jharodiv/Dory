import numpy as np
import tensorflow as tf
import pickle
import os
import config

def load_trained_model():
    # Load the trained model and label encoder
    model_path = os.path.join(config.MODEL_DIR, 'best_dory_model.h5')
    le_path = os.path.join(config.MODEL_DIR, 'label_encoder.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder not found at {le_path}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load label encoder
    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

def predict_single_sample(model, label_encoder, sample):

    #Predict class for a single audio sample
    # Ensure sample has correct shape
    if len(sample.shape) == 2:
        sample = np.expand_dims(sample, axis=0)  # Add batch dimension
        sample = np.expand_dims(sample, axis=-1)  # Add channel dimension
    elif len(sample.shape) == 3 and sample.shape[-1] != 1:
        sample = np.expand_dims(sample, axis=-1)  # Add channel dimension
    
    # Make prediction
    prediction = model.predict(sample, verbose=0)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]
    
    # Get class name
    predicted_class = label_encoder.classes_[predicted_class_idx]
    
    return predicted_class, confidence, prediction[0]

def predict_batch(model, label_encoder, samples):
    """Predict classes for a batch of audio samples"""
    # Ensure samples have correct shape
    if len(samples.shape) == 3:
        samples = np.expand_dims(samples, axis=-1)
    
    # Make predictions
    predictions = model.predict(samples, verbose=0)
    predicted_classes_idx = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Get class names
    predicted_classes = [label_encoder.classes_[idx] for idx in predicted_classes_idx]
    
    return predicted_classes, confidences, predictions

def main():
    try:
        # Load model and label encoder
        model, label_encoder = load_trained_model()
        print("✅ Model and label encoder loaded successfully!")
        print(f"Available classes: {list(label_encoder.classes_)}")
        
        # Example: Load some test data for demonstration
        # Replace this with your actual audio preprocessing pipeline
        if os.path.exists(config.X_PATH):
            X = np.load(config.X_PATH, allow_pickle=True)
            
            # Take a few samples for demonstration
            test_samples = X[:5]
            
            print(f"\n🔍 Making predictions on {len(test_samples)} samples...")
            
            # Single sample prediction
            for i, sample in enumerate(test_samples):
                predicted_class, confidence, _ = predict_single_sample(
                    model, label_encoder, sample
                )
                print(f"Sample {i+1}: {predicted_class} (confidence: {confidence:.3f})")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure to train the model first by running train.py")

if __name__ == "__main__":
    main()