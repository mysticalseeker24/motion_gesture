# Hand Gesture Recognition Backend

This backend system powers the advanced hand gesture recognition application, translating hand gestures to text and speech in real-time using computer vision, deep learning, and AI-powered interpretation.

## Features

- **Real-time Gesture Recognition**: Detects and interprets hand gestures from camera feed with fluid video processing
- **Text-to-Speech Conversion**: Converts recognized gestures into audible speech using multiple TTS engines
- **OpenAI Vision Integration**: Enhances gesture interpretations with sophisticated AI analysis
- **Two-Tier Processing**: Combines fast local detection with enhanced AI interpretation for optimal performance
- **Dynamic Performance Optimization**: Automatically adjusts processing parameters for smooth operation
- **RESTful API**: Provides endpoints for integration with the frontend application

## Architecture

The system consists of several key components:

1. **Gesture Recognition Engine**: 
   - Local PyTorch-based model for fast gesture detection
   - OpenAI Vision for enhanced gesture interpretation

2. **Text-to-Speech Engine**: Converts text interpretations to audio using:
   - Google Text-to-Speech (gTTS)
   - pyttsx3 (offline TTS)
   - OpenAI TTS (high-quality voice)

3. **FastAPI Backend**: Provides RESTful endpoints and facilitates communication with the frontend

4. **Concurrent Processing Pipeline**: Uses threading and queues for non-blocking operations

## Supported Gestures

The system recognizes the following gestures:
- "hello"
- "helpme" 
- "iloveyou"
- "namaste"
- "no"
- "searching hospital"
- "searching police"
- "thanks"
- "yes"

With OpenAI Vision integration, it can also interpret free-form gestures beyond these predefined classes.

## Prerequisites

- Python 3.9+
- OpenCV
- PyTorch
- OpenAI API key
- Compatible webcam or camera

## Installation

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file in the root directory):
```
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_TTS_MODEL=tts-1
VISION_ENABLED=true
IMAGE_CAPTURE_INTERVAL=5
```

## Usage

### Running the System

```bash
python main.py
```

### Command Line Arguments

```bash
# Use traditional model instead of OpenAI Vision
python main.py --no-vision

# Specify camera index (if multiple cameras are available)
python main.py --camera 1

# Set custom resolution
python main.py --width 1280 --height 720
```

## API Endpoints

- `GET /`: API information
- `GET /gestures`: List all supported gestures and meanings
- `POST /detect`: Detect gestures in an uploaded image
- `POST /text-to-speech`: Convert text to speech
- `POST /enhance-interpretation`: Enhance gesture interpretation using OpenAI

## Integration with Frontend

The backend is designed to work with the React Native frontend. See the [Technical Details](TECHNICAL_DETAILS.md) document for comprehensive integration instructions.

## License

MIT License
