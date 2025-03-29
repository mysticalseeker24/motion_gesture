import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# TensorFlow model paths (leveraging existing model from hackblitz)
TF_MODEL_PATH = os.path.join(MODEL_DIR, 'tf_gesture_model')

# PyTorch model paths
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, 'pytorch_gesture_model')

# Label map path
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'label_map.pbtxt')

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
OPENAI_TTS_MODEL = os.getenv('OPENAI_TTS_MODEL', 'tts-1')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 500))

# Vision-based Gesture Recognition Configuration
VISION_ENABLED = os.getenv('VISION_ENABLED', 'true').lower() == 'true'
IMAGE_CAPTURE_INTERVAL = float(os.getenv('IMAGE_CAPTURE_INTERVAL', 2.0))  # Seconds between frame captures for OpenAI analysis

# OpenAI API settings
OPENAI_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "text-davinci-003": "text-davinci-003",
    "text-davinci-002": "text-davinci-002",
    "text-curie-001": "text-curie-001",
    "text-babbage-001": "text-babbage-001",
    "text-ada-001": "text-ada-001"
}

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Gesture recognition settings
CONFIDENCE_THRESHOLD = 0.7

# TTS settings
TTS_ENGINE = "gtts"  # Options: 'gtts', 'pyttsx3', 'openai'
TTS_LANGUAGE = "en"
TTS_OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'audio')
OPENAI_TTS_VOICES = {
    "alloy": "alloy",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "shimmer": "shimmer"
}
OPENAI_TTS_VOICE = OPENAI_TTS_VOICES["nova"]

# Predefined gestures and their meanings
GESTURE_MEANINGS = {
    "hello": "Hello, how are you?",
    "helpme": "I need help please.",
    "iloveyou": "I love you.",
    "namaste": "Namaste, I greet you with respect.",
    "no": "No.",
    "searching hospital": "I need to find a hospital.",
    "searching police": "I need police assistance.",
    "thanks": "Thank you very much.",
    "yes": "Yes."
}
GESTURE_LABELS = ["hello", "helpme", "iloveyou", "namaste", "no", "searching hospital", "searching police", "thanks", "yes"]
