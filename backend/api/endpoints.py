from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import io
import base64
import time
import asyncio
import json
import os
from typing import List, Dict, Optional

from utils.gesture_recognition import GestureRecognizer
from utils.tts import get_tts_engine
from utils.openai_integration import OpenAIAssistant
from utils.audio_manager import audio_manager
from config.config import API_HOST, API_PORT, GESTURE_MEANINGS, TTS_OUTPUT_DIR

# Initialize the FastAPI app
app = FastAPI(title="Hand Gesture Recognition API", 
              description="API for recognizing hand gestures and converting them to text/speech",
              version="1.0.0")

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static audio files directory
app.mount("/audio", StaticFiles(directory=TTS_OUTPUT_DIR), name="audio")

# Create global instances of our core components
try:
    gesture_recognizer = GestureRecognizer(use_openai=True)
    tts_engine = get_tts_engine()
    openai_assistant = OpenAIAssistant()
    print("API services initialized successfully")
except Exception as e:
    print(f"Error initializing API services: {e}")
    gesture_recognizer = GestureRecognizer(use_openai=False)
    tts_engine = get_tts_engine()
    openai_assistant = None


# Websocket connection manager for real-time communication
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, message: Dict):
        await websocket.send_json(message)

    async def broadcast_json(self, message: Dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()


@app.get("/")
async def root():
    """Root endpoint providing API info"""
    return {"message": "Hand Gesture Recognition API", 
            "version": "1.0.0", 
            "status": "active"}


@app.get("/gestures")
async def list_gestures():
    """Get a list of all supported gestures and their meanings"""
    return JSONResponse(content=GESTURE_MEANINGS)


@app.post("/detect")
async def detect_gesture(file: UploadFile = File(...)):
    """Detect gestures in an uploaded image"""
    try:
        # Read the image file
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        # Process the image
        processed_image, detection = gesture_recognizer.process_frame(image)
        
        # If no gesture detected
        if detection is None:
            return JSONResponse(content={"success": False, "message": "No gesture detected"})
        
        # Get gesture details
        gesture_label = detection["label"]
        confidence = detection["confidence"]
        interpretation = gesture_recognizer._get_interpretation(gesture_label)
        
        # Generate audio for the interpretation
        audio_path = None
        audio_url = None
        
        # Check if we have a cached audio file
        audio_path = audio_manager.get_audio_path(interpretation)
        if not audio_path:
            # Generate speech if not cached
            audio_path = tts_engine.generate_speech(interpretation)
            audio_manager.save_audio(interpretation, audio_path)
        
        # Get URL for the audio file
        audio_url = audio_manager.get_audio_url(audio_path)
        
        # Convert processed image to base64 for response
        _, buffer = cv2.imencode(".jpg", processed_image)
        img_str = base64.b64encode(buffer).decode("utf-8")
        
        # Return the results
        return JSONResponse(content={
            "success": True,
            "gesture": gesture_label,
            "confidence": confidence,
            "interpretation": interpretation,
            "processed_image": img_str,
            "audio_url": audio_url
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Error: {str(e)}"})


@app.post("/text-to-speech")
async def convert_text_to_speech(background_tasks: BackgroundTasks, text: str):
    """Convert provided text to speech and return audio file"""
    try:
        # Check if we already have this text in cache
        cached_path = audio_manager.get_audio_path(text)
        if cached_path:
            return FileResponse(cached_path, media_type="audio/mpeg")
            
        # Convert text to speech if not cached
        filepath = tts_engine.speak(text)
        
        # Save to cache
        audio_manager.save_audio(text, filepath)
        
        # Return the file
        return FileResponse(filepath, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Error: {str(e)}"})


@app.get("/audio-url/{text}")
async def get_audio_url(text: str):
    """Get a URL to the audio file for the given text, generating if needed"""
    try:
        # Check if we already have this text in cache
        cached_path = audio_manager.get_audio_path(text)
        
        # Generate speech if not in cache
        if not cached_path:
            filepath = tts_engine.generate_speech(text)
            cached_path = audio_manager.save_audio(text, filepath)
        
        # Get the URL for the file
        audio_url = audio_manager.get_audio_url(cached_path)
        
        if not audio_url:
            raise HTTPException(status_code=404, detail="Failed to generate audio URL")
            
        return JSONResponse(content={
            "success": True,
            "audio_url": audio_url
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Error: {str(e)}"})


@app.get("/audio/{filename}")
async def get_audio_file(filename: str = Path(..., description="Audio filename")):
    """Get a specific audio file by name"""
    try:
        # Validate filename to prevent directory traversal
        if '..' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Construct full path
        file_path = os.path.join(TTS_OUTPUT_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
            
        return FileResponse(file_path, media_type="audio/mpeg")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        return JSONResponse(content={"success": False, "message": f"Error: {str(e)}"})


@app.post("/enhance-text")
async def enhance_text(text: str, context: Optional[str] = None):
    """Enhance text using OpenAI"""
    try:
        enhanced_text = openai_assistant.enhance_interpretation(text, context)
        return JSONResponse(content={
            "success": True, 
            "original_text": text, 
            "enhanced_text": enhanced_text
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Error: {str(e)}"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Websocket endpoint for real-time video processing"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive the frame as base64 string
            data = await websocket.receive_text()
            json_data = json.loads(data)
            base64_image = json_data.get("image")
            
            if not base64_image:
                continue
                
            # Convert base64 to image
            img_bytes = base64.b64decode(base64_image)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the frame for gesture detection
            processed_frame, detection = gesture_recognizer.process_frame(frame)
            
            # Prepare the response
            if detection:
                gesture_label = detection["label"]
                confidence = detection["confidence"]
                interpretation = gesture_recognizer._get_interpretation(gesture_label)
                
                # Generate audio for the interpretation
                audio_path = None
                audio_url = None
                
                # Check if we have a cached audio file
                audio_path = audio_manager.get_audio_path(interpretation)
                if not audio_path:
                    # Generate speech if not cached
                    audio_path = tts_engine.generate_speech(interpretation)
                    audio_manager.save_audio(interpretation, audio_path)
                
                # Get URL for the audio file
                audio_url = audio_manager.get_audio_url(audio_path)
                
                # Convert processed frame back to base64
                _, buffer = cv2.imencode(".jpg", processed_frame)
                processed_base64 = base64.b64encode(buffer).decode("utf-8")
                
                # Send the response
                await manager.send_json(websocket, {
                    "success": True,
                    "gesture": gesture_label,
                    "confidence": confidence,
                    "interpretation": interpretation,
                    "processed_image": processed_base64,
                    "audio_url": audio_url
                })
            else:
                # No detection
                _, buffer = cv2.imencode(".jpg", processed_frame)
                processed_base64 = base64.b64encode(buffer).decode("utf-8")
                
                await manager.send_json(websocket, {
                    "success": False,
                    "message": "No gesture detected",
                    "processed_image": processed_base64
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


def start_api():
    """Start the FastAPI server"""
    uvicorn.run("api.endpoints:app", host=API_HOST, port=API_PORT, reload=True)


if __name__ == "__main__":
    start_api()
