import os

dataDirection = os.path.expanduser("~/Document/Github/Dory")
xPath = os.path.join(dataDirection, "X.pny")
yPath = os.path.join(dataDirection, "y.pny")
modelDirection = os.path.join(dataDirection, "models")

#Training Parameters
batchSize = 16 # Optimized for 8GB RAM
learningRate = 0.0001
epochs = 100
randomSeed = 42

# Data Split Ratio
trainSplit = 0.7
valSplit = 0.15
testSplit = 0.15

patience = 15
lrPatience = 8
minLR = 1e-6
lrFactor = 0.5

# Class imbalance threshold
imbalanceThreshold = 3.0