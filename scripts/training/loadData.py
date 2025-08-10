import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU acceleration available: {len(physical_devices)} device(s)")
    try:
        #Enable memory growth to prevent allocation issues
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(f"Memory growth setting failed: {e}")
else:
    print("No GPU acceleration - USING CPU")


dataDirection = os.path.expanduser("~/Documents/Github/Dory/features")
xPath = os.path.join(dataDirection, "X.npy")
yPath = os.path.join(dataDirection, "y.npy")
modelDirection = os.path.join(dataDirection, "models")

#Create directories if they don't exist
os.makedirs(modelDirection, exist_ok=True)

print(f"\n Looking for data files")
print(f"X data{xPath}")
print(f"y data{yPath}")

#check if the data exist
if not os.path.exists(xPath):
    print(f"X.npy not found at {xPath}")
    print("Please update the path or copy your files")
    exit(1)

if not os.path.exists(yPath):
    print(f"y.npy not found at {yPath}")
    print("Please update the path or copy your files")
    exit(1)

#Load Data
print(f"loading training data")

X = np.load(xPath, allow_pickle=True)
y = np.load(yPath, allow_pickle=True)


print(f"Original Shaps -X: {X.shape}, y: {y.shape}")