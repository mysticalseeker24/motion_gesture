import cv2
import numpy as np
import time
from collections import deque

from config.config import CONFIDENCE_THRESHOLD, FRAME_WIDTH, FRAME_HEIGHT
from utils.model_loader import get_model
from utils.tts import get_tts_engine
from utils.openai_integration import OpenAIAssistant

class GestureRecognizer:
    """Main class for gesture recognition, processing, and interpretation"""
    
    def __init__(self, model_type="tensorflow", use_openai=True):
        # Initialize the model
        self.model = get_model(model_type)
        
        # Initialize TTS engine
        self.tts_engine = get_tts_engine()
        
        # Initialize OpenAI if enabled
        self.use_openai = use_openai
        if use_openai:
            try:
                self.openai = OpenAIAssistant()
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI: {e}")
                self.use_openai = False
        
        # Settings
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # State tracking
        self.last_spoken_time = 0
        self.speech_cooldown = 3  # seconds between speech outputs
        self.last_recognized_gesture = None
        self.gesture_history = deque(maxlen=10)  # Store last 10 gestures
        
        # Frame counter for periodic processing
        self.frame_counter = 0
        self.process_every_n_frames = 5  # Process every 5 frames for efficiency
    
    def process_frame(self, frame):
        """Process a video frame and detect gestures
        
        Args:
            frame: The video frame to process
            
        Returns:
            processed_frame: Frame with annotations
            detection_results: Dictionary with detection information
        """
        # Increment frame counter
        self.frame_counter += 1
        
        # Only process every Nth frame for efficiency
        if self.frame_counter % self.process_every_n_frames != 0:
            return frame, None
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Create a copy for annotations
        annotated_frame = frame.copy()
        
        # Run detection
        detections = self.model.predict(frame)
        
        if detections is None:
            # Mock detection if real model isn't available
            mock_detections = self._mock_detection(frame)
            if mock_detections is not None:
                detections = mock_detections
        
        if detections is None:
            return annotated_frame, None
        
        # Process results
        boxes = detections['boxes']
        classes = detections['classes']
        scores = detections['scores']
        
        # Keep track of the highest confidence detection
        max_score = 0
        best_detection = None
        
        # Process each detection
        for i in range(len(scores)):
            if scores[i] > self.confidence_threshold:
                if scores[i] > max_score:
                    max_score = scores[i]
                    class_id = classes[i]
                    label = self.model.label_map.get_name(class_id)
                    box = boxes[i]
                    best_detection = {
                        'label': label,
                        'confidence': float(scores[i]),
                        'box': box.tolist()
                    }
        
        # If we found a valid detection
        if best_detection:
            # Draw the detection on the frame
            self._annotate_frame(annotated_frame, best_detection)
            
            # Get the detected gesture label
            gesture_label = best_detection['label']
            
            # Update history
            if gesture_label != self.last_recognized_gesture:
                self.gesture_history.append(gesture_label)
                self.last_recognized_gesture = gesture_label
                
                # Speak the interpretation (with cooldown)
                current_time = time.time()
                if current_time - self.last_spoken_time > self.speech_cooldown:
                    self._speak_interpretation(gesture_label)
                    self.last_spoken_time = current_time
            
            return annotated_frame, best_detection
        
        return annotated_frame, None
    
    def _mock_detection(self, frame):
        """Mock detection for testing purposes"""
        # Simulate a detection with a random label and confidence
        import random
        from config.config import GESTURE_LABELS
        label = random.choice(GESTURE_LABELS)
        confidence = random.uniform(0.5, 1.0)
        
        # Simulate a bounding box
        h, w, _ = frame.shape
        xmin = random.uniform(0, 0.5)
        ymin = random.uniform(0, 0.5)
        xmax = random.uniform(0.5, 1.0)
        ymax = random.uniform(0.5, 1.0)
        box = [xmin, ymin, xmax, ymax]
        
        # Return mock detection
        return {
            'boxes': [box],
            'classes': [0],  # Class ID doesn't matter for mock detection
            'scores': [confidence]
        }
    
    def _annotate_frame(self, frame, detection):
        """Add visual annotations to frame showing the detection"""
        h, w, _ = frame.shape
        box = detection['box']
        label = detection['label']
        confidence = detection['confidence']
        
        # Convert normalized coordinates to pixel coordinates
        ymin, xmin, ymax, xmax = box
        xmin, xmax = int(xmin * w), int(xmax * w)
        ymin, ymax = int(ymin * h), int(ymax * h)
        
        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Draw label background
        text = f"{label}: {confidence:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, 
                      (xmin, ymin - text_size[1] - 5), 
                      (xmin + text_size[0], ymin), 
                      (0, 255, 0), 
                      -1)
        
        # Draw text
        cv2.putText(frame, text, (xmin, ymin - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add interpretation text at bottom of frame
        interpretation = self._get_interpretation(label)
        cv2.rectangle(frame, 
                      (0, h - 40), 
                      (w, h), 
                      (0, 0, 0), 
                      -1)
        cv2.putText(frame, interpretation, (10, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _get_interpretation(self, gesture_label):
        """Get the interpretation of the gesture using OpenAI if available"""
        if self.use_openai:
            if len(self.gesture_history) > 1:
                # Use contextual interpretation if we have history
                return self.openai.generate_contextual_response(
                    list(self.gesture_history), gesture_label
                )
            else:
                # Use simple enhancement for a single gesture
                return self.openai.enhance_interpretation(gesture_label)
        else:
            # Use basic dictionary meanings if OpenAI is not available
            from config.config import GESTURE_MEANINGS
            return GESTURE_MEANINGS.get(gesture_label, f"Detected: {gesture_label}")
    
    def _speak_interpretation(self, gesture_label):
        """Convert the interpretation to speech"""
        interpretation = self._get_interpretation(gesture_label)
        self.tts_engine.speak(interpretation)
