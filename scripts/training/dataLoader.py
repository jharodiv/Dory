import numpy as np
import config
from utils import (
    analyze_class_distribution, 
    prepare_labels, 
    calculate_class_weights, 
    split_data
)

def load_data():
    print(f"\n📊 Loading training data...")
    X = np.load(config.X_PATH, allow_pickle=True)
    y = np.load(config.Y_PATH, allow_pickle=True)
    
    print(f"Original shapes - X: {X.shape}, y: {y.shape}")
    return X, y

def preprocess_data(X, y):
    # Analyze class distribution
    class_counts, imbalance_ratio = analyze_class_distribution(y)
    
    # Prepare labels
    y_encoded, label_encoder = prepare_labels(y)
    
    # Calculate class weights
    class_weight_dict = calculate_class_weights(y_encoded)
    
    # Split data
    (X_train, X_val, X_test), (y_train, y_val, y_test) = split_data(X, y_encoded)
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'label_encoder': label_encoder,
        'class_weights': class_weight_dict,
        'class_counts': class_counts,
        'imbalance_ratio': imbalance_ratio
    }

def get_data():
    # Main function to get preprocessed data
    X, y = load_data()
    data_dict = preprocess_data(X, y)
    return data_dict