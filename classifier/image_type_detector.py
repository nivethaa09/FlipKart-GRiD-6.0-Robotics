import cv2
import numpy as np
from tensorflow import keras
import os

layers = keras.layers
models = keras.models

# First, let's create a small model for image type detection
def create_type_detection_model():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes: single, bunch, veggies
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Train and save the model with dummy data
def ensure_model_exists():
    model_path = os.path.join('models', 'image_type_model.h5')
    if not os.path.exists('models'):
        os.makedirs('models')
    
    if not os.path.exists(model_path):
        # Create and train model with dummy data
        model = create_type_detection_model()
        X_train = np.random.randn(100, 128, 128, 3)
        y_train = np.random.randint(0, 3, 100)
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        model.save(model_path)

# Main detection function
def detect_image_type(image_path):
    """
    Detect the type of image: single product, bunch of products, or vegetables
    Args:
        image_path: Path to the image file
    Returns:
        str: 'single', 'bunch', or 'veggies'
    """
    # Ensure model exists
    ensure_model_exists()
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Simple rule-based detection using image processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Count contours to determine if it's a bunch
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate color features
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        # Decision logic
        if len(contours) > 10:  # Multiple objects detected
            return 'bunch'
        elif green_ratio > 0.3:  # Significant green color
            return 'veggies'
        else:
            return 'single'
            
    except Exception as e:
        print(f"Error in detect_image_type: {str(e)}")
        return 'single'  

if __name__ == "__main__":
    # Test the detector
    test_image_path = "Test.jpg"
    if os.path.exists(test_image_path):
        result = detect_image_type(test_image_path)
        print(f"Detected image type: {result}")