import numpy as np
import sys
from utils import (
    configure_tensorflow,
    check_data_files,
    create_directories,
    plot_training_history,
    evaluate_model,
    save_model_files
)
from dataLoader import get_data
from model import build_and_compile_model, train_model

def main():
    # Configure TensorFlow
    has_gpu = configure_tensorflow()
    
    # Check if data files exist
    if not check_data_files():
        sys.exit(1)
    
    # Create necessary directories
    create_directories()
    
    # Load and preprocess data
    data = get_data()
    
    # Extract data
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    label_encoder = data['label_encoder']
    class_weight_dict = data['class_weights']
    
    # Build and compile model
    num_classes = len(np.unique(y_train))
    model = build_and_compile_model(X_train.shape[1:], num_classes)
    
    # Train model
    history = train_model(
        model, 
        X_train, y_train, 
        X_val, y_val, 
        class_weight_dict, 
        has_gpu
    )
    
    # Evaluate model
    test_accuracy = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Plot and save training history
    plot_path, history_path = plot_training_history(history)
    
    # Save model files
    import config
    le_path = f"{config.MODEL_DIR}/label_encoder.pkl"
    final_model_path = save_model_files(model, le_path, plot_path, history_path)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"Final test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()