import os

# Data paths
DATA_DIR = os.path.expanduser("~/Documents/GitHub/Dory/features")
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# Training parameters
BATCH_SIZE = 16  # Optimized for 8GB RAM
LEARNING_RATE = 0.001
EPOCHS = 100
RANDOM_SEED = 42

# Data split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Early stopping parameters
PATIENCE = 15
LR_PATIENCE = 8
MIN_LR = 1e-6
LR_FACTOR = 0.5

# Class imbalance threshold
IMBALANCE_THRESHOLD = 3.0