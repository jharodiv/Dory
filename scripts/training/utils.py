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
    #Checks if GPU (metal) is available
    physicalDevice = tf.config.list_physical_devices('GPU')
    if physicalDevice:
        print(f"GPU acceleration available: {len(physicalDevice)} devices")
        try:
            #Enable memory growth to prevent allocation issues
            for device in physicalDevice:
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as e:
            print(f"Memory growth setting failed: {e}")
    else:
        print("No GPU acceleration - using CPU")
    
    return len (physicalDevice) > 0

def checkDataFiles():
    if not os.path.exists(config.xPath):
        print(f"X.npy not found at {config.xPath}")
        print("Update the path")
        return False
    
    if not os.path.exists(config.yPath):
        print(f"y.npy not found at {config.yPath}")
        print("Update the path")
        return False
    
    return True

def createDirectories():
    os.makedirs(config.modelDirection, exist_ok=True)

def analyze_class_distribution(y):
    print(f"\n Class Distribution Analysis:")

    classCount = Counter(y)
    for className, count in classCount.items():
        percentage = (count / len(y)) * 100
        print(f"{className}: {count} samples ({percentage:.1f}%)")

    maxCount = max(classCount.values())
    minCount = min(classCount.values())
    imbalanceRatio = maxCount / minCount
    print(f"\n Class Imbalance Ratio: {imbalanceRatio:.2f}")
    if imbalanceRatio > config.imbalanceThreshold:
        print("Warning: Significant class imbalance detected")

    return classCount, imbalanceRatio

def prepareLabels(y):
    le = LabelEncoder()
    yEncoded = le.fit_transform(y)

    #Save label encoder
    lePath = os.path.join(config.modelDirection, "label_encoder.pkl")
    with open(lePath, 'wb') as f:
        pickle.dump(le, f)

    print(f"n\Label Mapping:")
    for i, className in enumerate(le.classes_):
        print(f"{i}: {className}")

    return yEncoded, le

def calculate_class_weight(yEncoded):
    classWeights = compute_class_weight(
        'balanced',
        classes=np.unique(yEncoded),
        y=yEncoded
    )
    classWeightDict = dict(enumerate(classWeights))
    print(f"\n class weights: {classWeightDict}")
    return classWeightDict

def splitData(X, yEncoded):
    #Shuffle Data
    np.random.seed(config.randomSeed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    yEncoded = yEncoded[indices]

    #split data
    trainSplit = int(config,trainSplit * len(X))
    valSplit = int((config.trainSplit + config.valSplit) * len(X))

    XTrain = X[:trainSplit]
    XVal = X[trainSplit:valSplit]
    XTest = X[valSplit]

    yTrain = yEncoded[:trainSplit]
    yVal = yEncoded[trainSplit:valSplit]
    yTest = yEncoded[valSplit]


    #Expand dims for CNN input
    XTrain = np.expand_dims(XTrain, axis = -1)
    XVal = np.expand_dims(XVal, axis = -1)
    XTest = np.expand_dims(XTest, axis = -1)

    print(f"\n Dataset splits")
    print(f" Train : X = {XTrain.shape}, y = {yTrain.shape}")
    print(f" Val : X = {XVal.shape}, y = {yVal.shape}")
    print(f" Test : X = {XTest.shape}, y = {yTest.shape}")


    return (XTrain,XVal,XTest), (yTrain,yVal,yTest)

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
    print(f"\n Final Evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Detailed predictions on test set
    test_predictions = model.predict(X_test)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    print(f"\n Per-class performance on test set:")
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