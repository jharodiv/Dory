import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import config

def configure_tensorflow():
    # Check if GPU (Metal) is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"✅ GPU acceleration available: {len(physical_devices)} device(s)")
        try:
            # Enable memory growth to prevent allocation issues
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")
    else:
        print("⚠️  No GPU acceleration - using CPU")
    
    return len(physical_devices) > 0

def check_data_files():
    print(f"\n📁 Looking for data files...")
    print(f"X data: {config.X_PATH}")
    print(f"y data: {config.Y_PATH}")
    
    if not os.path.exists(config.X_PATH):
        print(f"❌ X.npy not found at {config.X_PATH}")
        print("Please update the path or copy your files to ~/Documents/Dory/")
        return False
    
    if not os.path.exists(config.Y_PATH):
        print(f"❌ y.npy not found at {config.Y_PATH}")
        print("Please update the path or copy your files to ~/Documents/Dory/")
        return False
    
    return True

def create_directories():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

def analyze_class_distribution(y):
    print(f"\n📈 Class Distribution Analysis:")
    class_counts = Counter(y)
    for class_name, count in class_counts.items():
        percentage = (count / len(y)) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Check for class imbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\n Class imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > config.IMBALANCE_THRESHOLD:
        print("   ⚠️  WARNING: Significant class imbalance detected!")
    
    return class_counts, imbalance_ratio

def prepare_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save label encoder
    le_path = os.path.join(config.MODEL_DIR, "label_encoder.pkl")
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\n🏷️ Label mapping:")
    for i, class_name in enumerate(le.classes_):
        print(f"  {i}: {class_name}")
    
    return y_encoded, le

def calculate_class_weights(y_encoded):
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n Class weights: {class_weight_dict}")
    return class_weight_dict

def split_data(X, y_encoded):
    # Shuffle data
    np.random.seed(config.RANDOM_SEED)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y_encoded = y_encoded[indices]
    
    # Split data
    train_split = int(config.TRAIN_SPLIT * len(X))
    val_split = int((config.TRAIN_SPLIT + config.VAL_SPLIT) * len(X))
    
    X_train = X[:train_split]
    X_val = X[train_split:val_split]
    X_test = X[val_split:]
    
    y_train = y_encoded[:train_split]
    y_val = y_encoded[train_split:val_split]
    y_test = y_encoded[val_split:]
    
    # Expand dims for CNN input
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"\n📏 Dataset splits:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(config.MODEL_DIR, 'training_history.png')
    plt.savefig(plot_path)
    plt.show()
    
    # Save training history
    history_path = os.path.join(config.MODEL_DIR, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    return plot_path, history_path

def evaluate_model(model, X_test, y_test, label_encoder):
    print(f"\n🎯 Final Evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Detailed predictions on test set
    test_predictions = model.predict(X_test)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    print(f"\n📊 Per-class performance on test set:")
    for class_idx, class_name in enumerate(label_encoder.classes_):
        class_mask = (y_test == class_idx)
        if np.any(class_mask):
            class_accuracy = np.mean(test_pred_classes[class_mask] == y_test[class_mask])
            class_count = np.sum(class_mask)
            print(f"  {class_name}: {class_accuracy:.3f} accuracy ({class_count} samples)")
    
    return test_accuracy

def save_model_files(model, le_path, plot_path, history_path):
    final_model_path = os.path.join(config.MODEL_DIR, 'dory_model_final.h5')
    model.save(final_model_path)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Saved files:")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Best model: {os.path.join(config.MODEL_DIR, 'best_dory_model.h5')}")
    print(f"  - Label encoder: {le_path}")
    print(f"  - Training history: {history_path}")
    print(f"  - Training plot: {plot_path}")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Test your model with new audio samples")
    print(f"  2. If accuracy is still low, collect more training data")
    print(f"  3. Consider data augmentation if you have limited samples")
    
    return final_model_path