import cv2
import numpy as np
import time
import os
from pathlib import Path
from utils.openai_integration import OpenAIAssistant
from utils.pytorch_gesture_model import HandDetector, PyTorchGestureRecognizer
from config.config import FRAME_WIDTH, FRAME_HEIGHT, CONFIDENCE_THRESHOLD, IMAGE_CAPTURE_INTERVAL, GESTURE_LABELS
import threading
from queue import Queue
import concurrent.futures

class VisionGestureRecognizer:
    """Advanced gesture recognition system that combines local PyTorch-based gesture detection
    with OpenAI Vision enhancement for more accurate and contextual interpretation"""
    
    def __init__(self):
        # Optimization settings
        self.max_skip_frames = 2  # Process every Nth frame for better performance
        self.frame_counter = 0
        self.buffer_size = 10  # Number of frames to keep in buffer
        self.result_cache_size = 100  # Max size of result cache
        self.downscale_factor = 0.5  # Downscale frames for faster processing
        
        # Initialize thread pool for concurrent processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.processing_queue = Queue(maxsize=5)  # Limit queue size to prevent memory issues
        self.openai_results_queue = Queue()
        
        # Initialize components
        print("Initializing Vision-based gesture recognition...")
        self.openai = OpenAIAssistant()
        self.local_recognizer = PyTorchGestureRecognizer()
        self.hand_detector = HandDetector()
        
        # Frame processing parameters
        self.last_openai_processing_time = 0
        self.openai_processing_interval = IMAGE_CAPTURE_INTERVAL  # seconds
        
        # Gesture tracking variables
        self.last_gesture = None
        self.last_gesture_confidence = 0
        self.stable_gesture_counter = 0
        self.stable_gesture_threshold = 3  # Reduced for faster response
        self.current_stable_gesture = None
        
        # Result cache for quick lookups
        self.detection_cache = {}
        
        # Start background threads
        self._start_background_workers()
        
        print("Vision-based gesture recognition initialized")
    
    def _start_background_workers(self):
        """Start background worker threads for parallel processing"""
        # Start thread for processing frames in the background
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()
        
        # Start thread for OpenAI result processing
        self.openai_thread = threading.Thread(target=self._process_openai_results, daemon=True)
        self.openai_thread.start()
    
    def _background_processor(self):
        """Background thread for processing frames from queue"""
        while True:
            try:
                if not self.processing_queue.empty():
                    frame, frame_id = self.processing_queue.get(timeout=0.1)
                    
                    # Detect hands and process frame in background
                    result = self._process_frame_in_background(frame)
                    
                    # Cache the result with timestamp for expiration
                    self.detection_cache[frame_id] = {
                        'result': result,
                        'timestamp': time.time()
                    }
                    
                    # Clean up old cache entries
                    self._cleanup_cache()
                    
                    # Mark task as done
                    self.processing_queue.task_done()
                else:
                    # Sleep a bit to prevent CPU hogging
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Error in background processor: {e}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop
    
    def _process_openai_results(self):
        """Background thread for processing OpenAI results"""
        while True:
            try:
                if not self.openai_results_queue.empty():
                    result = self.openai_results_queue.get(timeout=0.1)
                    print(f"OpenAI enhanced interpretation: {result['detailed_analysis'][:100]}...")  # Preview
                    self.openai_results_queue.task_done()
                else:
                    # Sleep a bit to prevent CPU hogging
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing OpenAI results: {e}")
                time.sleep(0.1)  # Sleep on error to prevent tight loop
    
    def _cleanup_cache(self):
        """Clean up old entries from the detection cache"""
        # Remove old entries if cache is too large
        if len(self.detection_cache) > self.result_cache_size:
            # Get entries sorted by timestamp (oldest first)
            sorted_entries = sorted(self.detection_cache.items(), key=lambda x: x[1]['timestamp'])
            
            # Remove oldest entries
            entries_to_remove = len(self.detection_cache) - self.result_cache_size
            for i in range(entries_to_remove):
                if i < len(sorted_entries):
                    del self.detection_cache[sorted_entries[i][0]]
    
    def _process_frame_in_background(self, frame):
        """Process a frame in the background thread"""
        try:
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Downscale for faster processing if needed
            if self.downscale_factor < 1.0:
                h, w = frame_rgb.shape[:2]
                new_h, new_w = int(h * self.downscale_factor), int(w * self.downscale_factor)
                frame_rgb_small = cv2.resize(frame_rgb, (new_w, new_h))
            else:
                frame_rgb_small = frame_rgb
            
            # Detect hands
            hands = self.hand_detector.detect_hands(frame_rgb_small)
            
            # Process results
            detection_results = None
            
            if hands and len(hands) > 0:
                # Get first hand
                hand = hands[0]
                
                # Adjust coordinates if we downscaled
                if self.downscale_factor < 1.0:
                    x, y, w, h = hand['raw_box']
                    x = int(x / self.downscale_factor)
                    y = int(y / self.downscale_factor)
                    w = int(w / self.downscale_factor)
                    h = int(h / self.downscale_factor)
                    hand_box = (x, y, w, h)
                else:
                    hand_box = hand['raw_box']
                    x, y, w, h = hand_box
                
                # Extract hand region from original frame
                hand_img = frame_rgb[y:y+h, x:x+w]
                
                # Process if hand region is valid
                if hand_img.size > 0 and hand_img.shape[0] > 20 and hand_img.shape[1] > 20:
                    # Recognize gesture using local model
                    result = self.local_recognizer.predict(hand_img)
                    
                    # Process results
                    if result and 'classes' in result and len(result['classes']) > 0:
                        class_id = int(result['classes'][0])
                        confidence = float(result['scores'][0]) if 'scores' in result and len(result['scores']) > 0 else 0.0
                        
                        # Map class ID to gesture label
                        if 0 < class_id <= len(GESTURE_LABELS):
                            gesture_label = GESTURE_LABELS[class_id - 1]
                        else:
                            gesture_label = "unknown"
                    else:
                        gesture_label = "unknown"
                        confidence = 0.0
                    
                    # Create detection result
                    detection_results = {
                        "gesture_label": gesture_label,
                        "confidence": confidence,
                        "hand_position": hand_box,
                        "source": "local_model"
                    }
            
            return detection_results
            
        except Exception as e:
            print(f"Error in background frame processing: {e}")
            return None
    
    def process_frame(self, frame):
        """Process a video frame with optimized performance"""
        # Make a copy for annotation
        annotated_frame = frame.copy()
        
        # Generate a unique ID for this frame
        frame_id = f"frame_{int(time.time() * 1000)}_{self.frame_counter}"
        self.frame_counter += 1
        
        # Apply frame skipping to improve performance
        should_process = (self.frame_counter % (self.max_skip_frames + 1) == 0)
        
        detection_results = None
        
        if should_process:
            # Check if we already have this frame in cache (unlikely but possible)
            if frame_id in self.detection_cache:
                detection_results = self.detection_cache[frame_id]['result']
            else:
                # Add to processing queue if not full
                if not self.processing_queue.full():
                    try:
                        self.processing_queue.put((frame.copy(), frame_id), block=False)
                    except:
                        pass  # Queue full, skip this frame
        
        # Use the most recent detection result for annotation
        if not detection_results and self.detection_cache:
            # Get the most recent cached result
            sorted_cache = sorted(self.detection_cache.items(), 
                                 key=lambda x: x[1]['timestamp'], 
                                 reverse=True)
            if sorted_cache:
                detection_results = sorted_cache[0][1]['result']
        
        # Handle stable gesture tracking
        if detection_results and 'gesture_label' in detection_results:
            gesture_label = detection_results['gesture_label']
            confidence = detection_results.get('confidence', 0)
            
            # Update stability counter
            if gesture_label == self.last_gesture and confidence > CONFIDENCE_THRESHOLD:
                self.stable_gesture_counter += 1
                if self.stable_gesture_counter >= self.stable_gesture_threshold:
                    # We have a stable gesture - might trigger OpenAI analysis
                    if self.current_stable_gesture != gesture_label:
                        self.current_stable_gesture = gesture_label
                        
                        # Check if we should trigger OpenAI analysis
                        current_time = time.time()
                        if (current_time - self.last_openai_processing_time >= self.openai_processing_interval):
                            self.last_openai_processing_time = current_time
                            self._trigger_openai_analysis(frame.copy(), gesture_label)
            else:
                # Different gesture, reset counter
                self.stable_gesture_counter = 0
                if gesture_label != self.last_gesture:
                    self.current_stable_gesture = None
            
            # Update last gesture
            self.last_gesture = gesture_label
            self.last_gesture_confidence = confidence
            
            # Update detection results with stability information
            detection_results['is_stable'] = (self.stable_gesture_counter >= self.stable_gesture_threshold)
        
        # Annotate the frame with detection information
        annotated_frame = self._annotate_frame(annotated_frame, detection_results)
        
        return annotated_frame, detection_results
    
    def _annotate_frame(self, frame, detection_results):
        """Annotate the frame with detection information"""
        if not detection_results:
            # No detections, just show processing info
            cv2.putText(frame, f"Processing: {self.frame_counter}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Extract information
        gesture_label = detection_results.get('gesture_label', 'unknown')
        confidence = detection_results.get('confidence', 0)
        is_stable = detection_results.get('is_stable', False)
        hand_position = detection_results.get('hand_position')
        
        # Draw bounding box if we have hand position
        if hand_position:
            x, y, w, h = hand_position
            color = (0, 255, 0) if is_stable else (0, 255, 255)  # Green if stable, yellow if not
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Add text with gesture label and confidence
        label_text = f"{gesture_label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add stability indicator
        stability_color = (0, 255, 0) if is_stable else (0, 0, 255)
        stability_text = f"Stable: {self.stable_gesture_counter}/{self.stable_gesture_threshold}"
        cv2.putText(frame, stability_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, stability_color, 2)
        
        # Add OpenAI status
        time_since_last_openai = int(time.time() - self.last_openai_processing_time)
        ready_for_openai = time_since_last_openai >= self.openai_processing_interval
        openai_color = (0, 255, 0) if ready_for_openai else (0, 0, 255)
        openai_text = f"OpenAI: {time_since_last_openai}/{int(self.openai_processing_interval)}s"
        cv2.putText(frame, openai_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, openai_color, 2)
        
        # Add processing info (frame counter, queue sizes)
        processing_text = f"Queue: {self.processing_queue.qsize()}/{self.processing_queue.maxsize}"
        cv2.putText(frame, processing_text, (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def _trigger_openai_analysis(self, frame, gesture_label):
        """Trigger OpenAI analysis in a non-blocking way"""
        try:
            # Submit the task to thread pool
            self.executor.submit(self._process_with_openai, frame, gesture_label)
        except Exception as e:
            print(f"Error triggering OpenAI analysis: {e}")
    
    def _process_with_openai(self, frame, gesture_label):
        """Process a frame with OpenAI in a separate thread"""
        try:
            # Save image to send to OpenAI (use lossy compression for speed)
            temp_img_path = Path("data/temp_images")
            temp_img_path.mkdir(parents=True, exist_ok=True)
            img_path = temp_img_path / f"gesture_{int(time.time())}.jpg"
            
            # Use lower quality for faster saving
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Send to OpenAI for enhanced interpretation
            enhanced = self.openai.analyze_gesture_image(
                str(img_path),
                f"This image contains a person making a '{gesture_label}' hand gesture. Please provide a concise description of what this gesture means and how it's commonly used. Keep your response under 100 words."
            )
            
            # Queue the results for processing
            if enhanced:
                result = {
                    "gesture_label": gesture_label,
                    "detailed_analysis": enhanced,
                    "timestamp": time.time(),
                    "source": "openai_enhanced"
                }
                self.openai_results_queue.put(result, block=False)
            
            # Clean up temp file
            if img_path.exists():
                os.remove(img_path)
                
        except Exception as e:
            print(f"Error in OpenAI processing thread: {e}")
    
    def enhance_interpretation(self, gesture_label, frame=None):
        """Get enhanced interpretation from OpenAI"""
        image_data = None
        
        # If we have a frame, encode it for OpenAI
        if frame is not None:
            # Save frame for sending to OpenAI
            temp_image_path = Path("data/temp_images")
            temp_image_path.mkdir(parents=True, exist_ok=True)
            img_path = temp_image_path / f"enhance_{int(time.time())}.jpg"
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Read the image file
            with open(img_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Clean up temp file
            if img_path.exists():
                os.remove(img_path)
        
        # Get enhanced interpretation (with timeout to prevent blocking)
        future = self.executor.submit(self.openai.enhance_interpretation, gesture_label, image_data)
        try:
            return future.result(timeout=2.0)  # 2-second timeout
        except concurrent.futures.TimeoutError:
            print("Warning: OpenAI enhancement timed out, using basic interpretation")
            return {"enhanced_interpretation": f"Detected {gesture_label} gesture"}
        except Exception as e:
            print(f"Error in enhance_interpretation: {e}")
            return {"enhanced_interpretation": f"Detected {gesture_label} gesture"}

    def __del__(self):
        """Clean up resources when the object is destroyed"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass
