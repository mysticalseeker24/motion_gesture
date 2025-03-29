import os
import cv2
import numpy as np
from pathlib import Path

# Try importing TensorFlow, but provide fallbacks if not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Using PyTorch implementation.")
    TF_AVAILABLE = False

# Import PyTorch for our real model implementation
try:
    import torch
    import torchvision.transforms as transforms
    from utils.pytorch_gesture_model import PyTorchGestureRecognizer
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Some features may be limited.")
    TORCH_AVAILABLE = False

from config.config import TF_MODEL_PATH, PYTORCH_MODEL_PATH, LABEL_MAP_PATH, GESTURE_LABELS


class LabelMap:
    def __init__(self, label_map_path):
        self.label_map = self._load_label_map(label_map_path)
    
    def _load_label_map(self, label_map_path):
        """Load label map from .pbtxt file"""
        label_map = {}
        try:
            with open(label_map_path, 'r') as f:
                current_id = None
                current_name = None
                for line in f:
                    line = line.strip()
                    if line.startswith('id:'):
                        current_id = int(line.split(':')[1].strip())
                    elif line.startswith('name:'):
                        current_name = line.split(':')[1].strip().strip('"')
                    
                    if current_id is not None and current_name is not None:
                        label_map[current_id] = current_name
                        current_id = None
                        current_name = None
            
            # If we couldn't load any labels, use the GESTURE_LABELS as fallback
            if not label_map and GESTURE_LABELS:
                for i, label in enumerate(GESTURE_LABELS):
                    label_map[i+1] = label
                print(f"Using {len(label_map)} labels from GESTURE_LABELS as fallback")
        except Exception as e:
            print(f"Error loading label map from {label_map_path}: {e}")
            # Fall back to GESTURE_LABELS as a last resort
            if GESTURE_LABELS:
                for i, label in enumerate(GESTURE_LABELS):
                    label_map[i+1] = label
                print(f"Using {len(label_map)} labels from GESTURE_LABELS due to error")
            
        return label_map
    
    def get_name(self, class_id):
        """Get class name from ID"""
        if class_id in self.label_map:
            return self.label_map[class_id]
        elif class_id >= 1 and class_id <= len(GESTURE_LABELS):
            # Fallback to GESTURE_LABELS if the ID is within range
            return GESTURE_LABELS[class_id-1]
        else:
            return "Unknown"


class TensorFlowModelLoader:
    def __init__(self, model_path, label_map_path):
        self.label_map = LabelMap(label_map_path)
        
        if not TF_AVAILABLE and TORCH_AVAILABLE:
            print("Using PyTorch-based gesture recognition model instead of TensorFlow")
            self.model = PyTorchGestureRecognizer()
            self.using_pytorch = True
        else:
            self.model = self._load_model(model_path)
            self.using_pytorch = False
    
    def _load_model(self, model_path):
        """Load saved TensorFlow model"""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Using PyTorch model.")
            return PyTorchGestureRecognizer() if TORCH_AVAILABLE else MockModel()
        
        try:
            model = tf.saved_model.load(model_path)
            if not model.signatures:
                raise Exception("Model has no signatures")
            return model
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            return None
    
    def predict(self, image):
        """Run inference on an image"""
        if self.model is None:
            return None
        
        # If we're using the PyTorch model, call it directly
        if self.using_pytorch or isinstance(self.model, PyTorchGestureRecognizer):
            return self.model.predict(image)
            
        # For TensorFlow model or mock model
        if isinstance(self.model, MockModel):
            return self.model.predict(image)
            
        # Original TensorFlow implementation
        if TF_AVAILABLE:
            # Prepare input tensor
            input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)
            
            # Run inference
            detections = self.model.signatures['serving_default'](input_tensor)
            
            # Process results
            boxes = detections['detection_boxes'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(np.int32)
            scores = detections['detection_scores'][0].numpy()
            
            return {
                'boxes': boxes,
                'classes': classes,
                'scores': scores,
                'num_detections': len(scores)
            }
        
        return None


class MockModel:
    """Mock TensorFlow model for gesture recognition"""
    def __init__(self):
        self.signatures = {'serving_default': self.serving_default}
        print("Initialized mock TensorFlow model for gesture recognition")
        
    def serving_default(self, input_tensor):
        """Mock serving_default function that mimics TensorFlow model inference"""
        # Extract image from tensor (simulating what TF would do)
        image = input_tensor.numpy()[0] if hasattr(input_tensor, 'numpy') else input_tensor[0]
        
        # Get image dimensions
        h, w = image.shape[:2] if len(image.shape) > 2 else (100, 100)
        
        # Mock detection with random position but realistic structure
        import random
        
        # Simulating detection of a random gesture from our label list
        class_id = random.randint(1, len(GESTURE_LABELS))
        confidence = random.uniform(0.7, 0.95)  # Reasonably high confidence
        
        # Create a bounding box in the center-ish area of the image
        # Format: [ymin, xmin, ymax, xmax] with normalized coordinates
        center_x, center_y = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
        width, height = random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)
        box = [
            max(0, center_y - height/2),  # ymin
            max(0, center_x - width/2),   # xmin
            min(1.0, center_y + height/2), # ymax
            min(1.0, center_x + width/2)   # xmax
        ]
        
        # Create a detection result in the format expected by the TensorFlow Object Detection API
        # Create numpy arrays that match the expected shapes and types
        import numpy as np
        
        # Convert to proper structure with batch dimension
        boxes_tensor = np.array([[box]], dtype=np.float32)  # Shape: [1, 1, 4]
        classes_tensor = np.array([[class_id]], dtype=np.int32)  # Shape: [1, 1]
        scores_tensor = np.array([[confidence]], dtype=np.float32)  # Shape: [1, 1]
        num_detections_tensor = np.array([1], dtype=np.int32)  # Shape: [1]
        
        # Structure the result like a TensorFlow model would
        class TensorLike:
            def __init__(self, data):
                self.data = data
                
            def numpy(self):
                return self.data
        
        # Return in a dictionary format matching TensorFlow's output
        return {
            'detection_boxes': TensorLike(boxes_tensor),
            'detection_classes': TensorLike(classes_tensor),
            'detection_scores': TensorLike(scores_tensor),
            'num_detections': TensorLike(num_detections_tensor)
        }
    
    def predict(self, image):
        """Direct prediction interface for compatibility with non-TensorFlow code"""
        # This provides an alternate interface that directly returns the processed results
        import numpy as np
        
        # Mock detection with random position but realistic structure
        import random
        
        # Simulating detection of a random gesture from our label list
        class_id = random.randint(1, len(GESTURE_LABELS))
        confidence = random.uniform(0.7, 0.95)  # Reasonably high confidence
        
        # Create a bounding box in the center-ish area of the image
        # Format: [ymin, xmin, ymax, xmax] with normalized coordinates
        center_x, center_y = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
        width, height = random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)
        box = [
            max(0, center_y - height/2),  # ymin
            max(0, center_x - width/2),   # xmin
            min(1.0, center_y + height/2), # ymax
            min(1.0, center_x + width/2)   # xmax
        ]
        
        # Return in the format expected by our application
        return {
            'boxes': np.array([box]),
            'classes': np.array([class_id]),
            'scores': np.array([confidence]),
            'num_detections': 1
        }


def copy_original_model(source_model_dir, target_model_dir):
    """Copy the original TensorFlow model from hackblitz to the backend models directory"""
    # This function would handle copying the model files from the original location
    # Since it involves file operations, we'll need to implement it carefully
    # For this template, we'll just create a placeholder
    os.makedirs(target_model_dir, exist_ok=True)
    print(f"Model would be copied from {source_model_dir} to {target_model_dir}")
    
    # In a real implementation, we would use shutil.copytree or similar
    # shutil.copytree(source_model_dir, target_model_dir)


def get_model(model_type="tensorflow"):
    """Factory function to get the appropriate model"""
    if model_type == "tensorflow":
        return TensorFlowModelLoader(TF_MODEL_PATH, LABEL_MAP_PATH)
    elif model_type == "pytorch":
        # Placeholder for PyTorch model loader
        # Would implement similar functionality as TensorFlowModelLoader
        pass
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
