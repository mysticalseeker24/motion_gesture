import cv2
import numpy as np
from utils.model_loader import TensorFlowModelLoader
from config.config import TF_MODEL_PATH, LABEL_MAP_PATH, GESTURE_LABELS

# Create a test script to validate our TensorFlow-based gesture recognition model

def test_gesture_model():
    print("Testing gesture recognition model...")
    print(f"Model path: {TF_MODEL_PATH}")
    print(f"Label map path: {LABEL_MAP_PATH}")
    print(f"Available gesture labels: {GESTURE_LABELS}")
    
    # Load the model
    model = TensorFlowModelLoader(TF_MODEL_PATH, LABEL_MAP_PATH)
    
    # Create a test image with a hand-like shape
    test_image = np.ones((320, 320, 3), dtype=np.uint8) * 255  # White background
    
    # Add a simple hand-like shape
    # Draw palm
    cv2.circle(test_image, (160, 180), 50, (200, 200, 200), -1)
    
    # Draw fingers
    for i in range(5):
        angle = np.pi / 2 + (i - 2) * np.pi / 10
        length = 80
        end_x = int(160 + length * np.cos(angle))
        end_y = int(180 - length * np.sin(angle))
        cv2.line(test_image, (160, 180), (end_x, end_y), (200, 200, 200), 15)
    
    # Save the test image
    cv2.imwrite("test_hand.jpg", test_image)
    print("Created test hand image as 'test_hand.jpg'")
    
    # Run inference
    print("\nRunning inference on test image...")
    result = model.predict(test_image)
    
    if result is None:
        print("Model returned None. No hands detected or processing error.")
        return
    
    # Print detection results
    print("\nDetection Results:")
    print(f"Number of detections: {result['num_detections']}")
    
    # Process all detections
    for i in range(len(result['scores'])):
        score = result['scores'][i]
        class_id = result['classes'][i]
        box = result['boxes'][i]
        
        label = model.label_map.get_name(class_id)
        
        print(f"Detection {i+1}: {label} (ID: {class_id})")
        print(f"Confidence: {score:.2f}")
        print(f"Bounding box: {box}")
        
        # Draw detection on image
        img_height, img_width = test_image.shape[:2]
        ymin, xmin, ymax, xmax = box
        
        # Convert normalized coordinates to pixels
        xmin = int(xmin * img_width)
        xmax = int(xmax * img_width)
        ymin = int(ymin * img_height)
        ymax = int(ymax * img_height)
        
        # Draw bounding box and label
        cv2.rectangle(test_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(test_image, f"{label}: {score:.2f}", 
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the annotated image
    cv2.imwrite("test_detection_result.jpg", test_image)
    print("\nAnnotated image saved as 'test_detection_result.jpg'")

if __name__ == "__main__":
    test_gesture_model()
