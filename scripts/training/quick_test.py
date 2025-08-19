import numpy as np
from tensorflow.keras.models import load_model

# Load model
model_path = "/Users/jemharoldcentino/Documents/GitHub/Dory/features/models/best_dory_model.h5"
model = load_model(model_path)

# Load data
X = np.load("/Users/jemharoldcentino/Documents/GitHub/Dory/features/X.npy")
y = np.load("/Users/jemharoldcentino/Documents/GitHub/Dory/features/y.npy", allow_pickle=True)  # allow strings

# Map integer predictions to class names
label_map = {0: "noise", 1: "not_wake", 2: "wake"}

# Loop through samples
for i in range(len(X)):
    sample = X[i]
    sample = np.expand_dims(sample, axis=(0, -1))
    
    pred_probs = model.predict(sample, verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    pred_label = label_map[pred_class]
    
    true_label = y[i]  # already a string
    
    print(f"Sample {i}: Predicted = {pred_label} | True = {true_label} | Probs = {pred_probs}")
