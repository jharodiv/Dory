import tensorflow as tf
from keras import layers, models
import os
import config

def create_optimized_model(input_shape, num_classes):

    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.MIN_LR
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'best_dory_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val, class_weight_dict, has_gpu=False):

    callbacks = create_callbacks()
    
    print(f"\n🚀 Starting training...")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Using {'GPU' if has_gpu else 'CPU'} acceleration")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def build_and_compile_model(input_shape, num_classes):

    model = create_optimized_model(input_shape, num_classes)
    model = compile_model(model)
    
    print(f"\n🏗️ Model Architecture:")
    model.summary()
    
    return model