# Dory Audio Classification Project

A modular machine learning project for audio classification using TensorFlow/Keras, optimized for macOS systems.

## Project Structure

```
dory-audio-classification/
├── config.py              # Configuration parameters
├── utils.py               # Utility functions
├── data_loader.py         # Data loading and preprocessing
├── model.py               # Model architecture and training functions
├── train.py               # Main training script
├── inference.py           # Inference and prediction functions
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Files Overview

### `config.py`
Contains all configuration parameters including:
- Data paths and directories
- Training hyperparameters
- Model architecture settings
- File paths for saved models

### `utils.py`
Utility functions for:
- TensorFlow GPU configuration
- Data file validation
- Class distribution analysis
- Data splitting and preprocessing
- Training history visualization
- Model evaluation and saving

### `data_loader.py`
Handles data loading and preprocessing:
- Loading numpy arrays from disk
- Label encoding and class weight calculation
- Train/validation/test splitting
- Data shape preparation for CNN

### `model.py`
Contains model-related functions:
- CNN architecture optimized for 8GB RAM
- Model compilation with optimizers
- Training callbacks setup
- Complete model building pipeline

### `train.py`
Main training script that orchestrates:
- Environment setup and validation
- Data loading and preprocessing
- Model creation and training
- Evaluation and result saving

### `inference.py`
Prediction and inference utilities:
- Loading trained models
- Single sample prediction
- Batch prediction
- Example usage demonstrations

## Setup and Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your data**:
   - Place your `X.npy` and `y.npy` files in `~/Documents/Dory/`
   - Or modify the `DATA_DIR` path in `config.py`

4. **Adjust configuration** (if needed):
   - Edit `config.py` to modify paths or hyperparameters
   - The default settings are optimized for MacBook Air 8GB RAM

## Usage

### Training a Model

Run the main training script:
```bash
python train.py
```

This will:
- Load and preprocess your data
- Create and train a CNN model
- Save the best model and training artifacts
- Display training progress and final evaluation

### Making Predictions

Use the inference script:
```bash
python inference.py
```

Or import in your own code:
```python
from inference import load_trained_model, predict_single_sample

model, label_encoder = load_trained_model()
prediction, confidence, _ = predict_single_sample(model, label_encoder, audio_sample)
```

### Customizing the Configuration

Edit `config.py` to modify:
- **Data paths**: Change `DATA_DIR` to point to your data location
- **Training parameters**: Adjust `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`
- **Model settings**: Modify split ratios, callbacks, etc.

## Output Files

After training, you'll find these files in your model directory:
- `best_dory_model.h5` - Best model during training
- `dory_model_final.h5` - Final trained model
- `label_encoder.pkl` - Label encoder for class names
- `training_history.pkl` - Training metrics history
- `training_history.png` - Training curves visualization

## System Requirements

- **macOS** (optimized for Apple Silicon with Metal GPU support)
- **8GB+ RAM** (batch size optimized for 8GB systems)
- **Python 3.8+**
- **TensorFlow 2.10+**

## Features

- 🍎 **macOS optimized**: Automatic GPU (Metal) detection and configuration
- 💾 **Memory efficient**: Optimized for 8GB RAM systems
- ⚖️ **Class imbalance handling**: Automatic class weight calculation
- 📊 **Comprehensive logging**: Detailed progress and evaluation metrics
- 🎯 **Modular design**: Easy to modify and extend
- 📈 **Visualization**: Automatic training history plots
- 🔍 **Easy inference**: Simple prediction interface

## Troubleshooting

### Common Issues

1. **GPU not detected**: Make sure you have TensorFlow Metal plugin installed
2. **Out of memory errors**: Reduce `BATCH_SIZE` in `config.py`
3. **Data files not found**: Check and update paths in `config.py`
4. **Poor performance**: Consider collecting more training data or data augmentation

### Performance Tips

- Ensure your data is properly preprocessed and normalized
- Monitor for class imbalance warnings
- Use early stopping to prevent overfitting
- Consider data augmentation for small datasets

## License

This project is open source. Feel free to modify and distribute according to your needs.
