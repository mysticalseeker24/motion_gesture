import os
import sys
import argparse
import cv2
import time
from threading import Thread
import threading

from utils.gesture_recognition import GestureRecognizer
from utils.tts import get_tts_engine
from utils.vision_gesture_recognition import VisionGestureRecognizer
from api.endpoints import start_api
from config.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, VISION_ENABLED, CONFIDENCE_THRESHOLD

# Initialize global variables
running = True


class VideoStream:
    """Class to handle video streaming from camera or video file"""
    def __init__(self, src=CAMERA_INDEX):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
    
    def start(self):
        """Start the thread for video capturing"""
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        """Continuously capture frames from the video source"""
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                return
                
            self.grabbed, self.frame = self.stream.read()
    
    def read(self):
        """Return the most recent frame"""
        return self.frame
    
    def stop(self):
        """Stop the video stream"""
        self.stopped = True
        self.stream.release()


def run_camera_feed(use_openai=True, camera_index=CAMERA_INDEX, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
    """Process video from camera feed for gesture recognition"""
    
    # Initialize TTS engine
    tts_engine = get_tts_engine()
    
    # Create output directories if they don't exist
    os.makedirs("data/audio", exist_ok=True)
    
    # Start API server
    print("Starting API server...")
    threading.Thread(target=start_api, daemon=True).start()
    
    # Choose the appropriate gesture recognition system
    try:
        if use_openai:
            recognizer = VisionGestureRecognizer()
            print("Using OpenAI Vision-based gesture recognition")
        else:
            recognizer = GestureRecognizer()
            print("Using traditional gesture recognition")
            
        # Initialize the camera with the best available settings
        cap = cv2.VideoCapture(camera_index)
        
        # Try to enable high FPS mode (30+ fps if possible)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        # Check camera format options for better performance
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG format for better performance
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened: {camera_index}, Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
        
        # Initialize variables for tracking gestures and performance
        last_reported_gesture = None
        last_report_time = 0
        gesture_cooldown = 1.5  # reduced from 2.0 seconds between gesture reports
        
        # Performance monitoring
        frame_times = []
        start_time = time.time()
        frames_processed = 0
        display_fps = 0
        last_fps_update = start_time
        
        try:
            while True:
                loop_start = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                try:
                    # Process frame for gesture recognition
                    annotated_frame, detection_results = recognizer.process_frame(frame)
                    
                    # Calculate FPS
                    frames_processed += 1
                    frame_times.append(time.time() - loop_start)
                    if len(frame_times) > 30:  # Keep only last 30 frames for calculating FPS
                        frame_times.pop(0)
                    
                    # Update FPS counter every second
                    if time.time() - last_fps_update > 1.0:
                        display_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
                        last_fps_update = time.time()
                    
                    # Add FPS information to frame
                    cv2.putText(annotated_frame, f"FPS: {display_fps:.1f}", 
                               (annotated_frame.shape[1] - 120, annotated_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Check if we have a detection to act upon
                    current_time = time.time()
                    if detection_results and 'gesture_label' in detection_results:
                        # Get gesture information
                        current_gesture = detection_results.get("gesture_label")
                        confidence = detection_results.get("confidence", 0)
                        is_stable = detection_results.get("is_stable", False)
                        
                        # Only process if it's stable and either a new gesture or enough time has passed
                        if (is_stable and 
                            confidence > CONFIDENCE_THRESHOLD and
                            (current_gesture != last_reported_gesture or 
                             current_time - last_report_time > gesture_cooldown)):
                            
                            print(f"Detected gesture: {current_gesture} (confidence: {confidence:.2f})")
                            
                            # Get the interpretation text
                            if 'detailed_analysis' in detection_results and detection_results['detailed_analysis']:
                                # Use the detailed analysis if available
                                interpretation = detection_results['detailed_analysis']
                            else:
                                # Use basic interpretation
                                interpretation = f"Detected {current_gesture} gesture"
                            
                            print(f"Interpretation: {interpretation}")
                            
                            # Generate speech from interpretation
                            audio_file = f"data/audio/gesture_{int(current_time)}.mp3"
                            threading.Thread(
                                target=lambda: tts_engine.generate_speech(interpretation, audio_file) if tts_engine else None,
                                daemon=True
                            ).start()
                            
                            # Update tracking variables
                            last_reported_gesture = current_gesture
                            last_report_time = current_time
                    
                    # Display the processed frame with low latency
                    cv2.imshow("Gesture Recognition", annotated_frame)
                except Exception as e:
                    import traceback
                    print(f"Error processing frame: {e}")
                    print(traceback.format_exc())
                
                # Check for exit key (use smallest wait time for better responsiveness)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Dynamically adjust frame skipping based on performance
                frame_time = time.time() - loop_start
                if hasattr(recognizer, 'max_skip_frames') and frame_time > 0.033:  # More than 30ms per frame
                    # Increase skip frames if processing is too slow
                    if recognizer.max_skip_frames < 5:  # Cap at 5 to ensure some processing happens
                        recognizer.max_skip_frames += 1
                        print(f"Increased frame skipping to {recognizer.max_skip_frames} for better performance")
                elif hasattr(recognizer, 'max_skip_frames') and frame_time < 0.02 and recognizer.max_skip_frames > 0:  # Less than 20ms
                    # Decrease skip frames if processing is fast enough
                    recognizer.max_skip_frames -= 1
                    print(f"Decreased frame skipping to {recognizer.max_skip_frames} for better quality")
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print("Camera feed stopped")
            
            # Print performance summary
            end_time = time.time()
            total_time = end_time - start_time
            avg_fps = frames_processed / total_time if total_time > 0 else 0
            print(f"Performance summary: Processed {frames_processed} frames in {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
    except Exception as e:
        import traceback
        print(f"Error initializing system: {e}")
        print(traceback.format_exc())


def copy_models():
    """Copy the models from the original hackblitz directory to our backend folder"""
    from shutil import copytree, copy2
    import os
    
    try:
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        
        # Copy label map
        print("Copying label map...")
        source_label_map = "../hackblitz/label_map.pbtxt"
        target_label_map = "models/label_map.pbtxt"
        copy2(source_label_map, target_label_map)
        
        # In a real scenario, we would also copy the TensorFlow model
        # This could be a more complex operation depending on how the model is stored
        # For example:
        # source_model = "../hackblitz/Tensorflow/workspace/models/my_ssd_mobnet/"
        # target_model = "models/tf_gesture_model/"
        # copytree(source_model, target_model)
        
        print("Model files copied successfully")
        return True
    
    except Exception as e:
        print(f"Error copying models: {e}")
        return False


def main():
    """Main entry point of the application"""
    parser = argparse.ArgumentParser(description="Gesture Recognition Server")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH, help="Frame width")
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT, help="Frame height")
    parser.add_argument("--use-vision", type=bool, default=VISION_ENABLED, help="Use OpenAI Vision for gesture recognition")
    parser.add_argument("--api-only", action="store_true", help="Only start the API server without camera feed")
    args = parser.parse_args()
    
    # Check if we need to copy models
    if not os.path.exists("models/label_map.pbtxt"):
        print("First run detected, copying model files...")
        if not copy_models():
            print("Error copying models, exiting")
            return
    
    # If no specific mode is specified, run both API and camera
    if not args.api_only:
        run_camera_feed(use_openai=args.use_vision, camera_index=args.camera, frame_width=args.width, frame_height=args.height)
    
    # If only API is running, keep the main thread alive
    if args.api_only:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting due to keyboard interrupt")


if __name__ == "__main__":
    main()
