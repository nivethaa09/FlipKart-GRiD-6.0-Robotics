import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

layers = keras.layers
models = keras.models

def create_small_freshness_model():
    model = models.Sequential([
        # Input layer - matches your 224x224x3 requirement
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Small feature extraction
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and save a pre-trained model with dummy data
def create_and_save_pretrained_model():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Create model
    model = create_small_freshness_model()
    
    # Create dummy training data (100 samples)
    X_train = np.random.randn(100, 224, 224, 3)
    y_train = np.random.randint(0, 2, 100)
    
    # Train model
    model.fit(
        X_train, 
        y_train,
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    model_path = 'models/freshness_model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    create_and_save_pretrained_model()