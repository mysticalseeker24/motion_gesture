import os
import time
import gtts
import pyttsx3
from pathlib import Path

from config.config import TTS_ENGINE, TTS_LANGUAGE, TTS_OUTPUT_DIR, OPENAI_API_KEY, OPENAI_TTS_VOICE


class TTSEngine:
    """Base class for Text-to-Speech engines"""
    def speak(self, text):
        """Convert text to speech"""
        raise NotImplementedError("Subclasses must implement speak()")
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text and save to file"""
        raise NotImplementedError("Subclasses must implement generate_speech()")


class GTTSEngine(TTSEngine):
    """Google Text-to-Speech engine"""
    def __init__(self, language=TTS_LANGUAGE, output_dir=TTS_OUTPUT_DIR):
        self.language = language
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def speak(self, text):
        """Convert text to speech using Google TTS"""
        try:
            # Generate a unique filename based on timestamp
            filename = f"tts_{int(time.time())}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create TTS object and save to file
            tts = gtts.gTTS(text=text, lang=self.language, slow=False)
            tts.save(filepath)
            
            # Play the audio file (platform-specific)
            self._play_audio(filepath)
            
            return filepath
        except Exception as e:
            print(f"GTTSEngine error: {e}")
            return None
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text and save to specified file"""
        try:
            # Use provided output file or generate a default one
            filepath = output_file or os.path.join(self.output_dir, f"tts_{int(time.time())}.mp3")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create TTS object and save to file
            tts = gtts.gTTS(text=text, lang=self.language, slow=False)
            tts.save(filepath)
            
            print(f"Speech generated and saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"GTTSEngine error: {e}")
            return None
    
    def _play_audio(self, filepath):
        """Play the audio file using platform-specific methods"""
        try:
            # Using playsound library (cross-platform)
            # Note: In a real implementation, we'd need to handle platform differences
            import playsound
            playsound.playsound(filepath)
        except ImportError:
            print("'playsound' package not installed. Cannot play audio.")
        except Exception as e:
            print(f"Error playing audio: {e}")


class PyttsxEngine(TTSEngine):
    """pyttsx3 Text-to-Speech engine (offline TTS)"""
    def __init__(self):
        self.engine = pyttsx3.init()
        # Configure voice properties
        self.engine.setProperty('rate', 150)  # Speed
        self.engine.setProperty('volume', 0.9)  # Volume (0 to 1)
        
        # Try to set a female voice if available
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak(self, text):
        """Convert text to speech using pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"PyttsxEngine error: {e}")
            return False
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text and save to specified file"""
        try:
            # Use provided output file or generate a default one
            if not output_file:
                output_dir = TTS_OUTPUT_DIR
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"tts_{int(time.time())}.wav")
            else:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Create an instance of the engine
            engine = pyttsx3.init()
            
            # Save to file
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            
            print(f"Speech generated and saved to {output_file}")
            return output_file
        except Exception as e:
            print(f"PyttsxEngine error: {e}")
            return None


class OpenAITTSEngine(TTSEngine):
    """OpenAI Text-to-Speech engine"""
    def __init__(self, api_key=OPENAI_API_KEY, voice=OPENAI_TTS_VOICE):
        self.api_key = api_key
        self.voice = voice
        self.output_dir = TTS_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def speak(self, text):
        """Convert text to speech using OpenAI TTS"""
        try:
            # Generate a unique filename based on timestamp
            filename = f"tts_{int(time.time())}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            # Generate audio file
            self._generate_audio(text, filepath)
            
            # Play the audio file
            self._play_audio(filepath)
            
            return filepath
        except Exception as e:
            print(f"OpenAITTSEngine error: {e}")
            return None
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text and save to specified file"""
        try:
            # Use provided output file or generate a default one
            filepath = output_file or os.path.join(self.output_dir, f"tts_{int(time.time())}.mp3")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Generate audio file
            self._generate_audio(text, filepath)
            
            print(f"Speech generated and saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"OpenAITTSEngine error: {e}")
            return None
    
    def _generate_audio(self, text, filepath):
        """Generate audio file using OpenAI TTS API"""
        try:
            # Import the required library
            from openai import OpenAI
            
            # Initialize the client
            client = OpenAI(api_key=self.api_key)
            
            # Generate the audio using OpenAI TTS API
            response = client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text
            )
            
            # Save the audio content to a file
            response.stream_to_file(filepath)
            
        except Exception as e:
            print(f"OpenAITTSEngine error: {e}")
    
    def _play_audio(self, filepath):
        """Play the audio file using platform-specific methods"""
        try:
            # Using playsound library (cross-platform)
            import playsound
            playsound.playsound(filepath)
        except ImportError:
            print("'playsound' package not installed. Cannot play audio.")
        except Exception as e:
            print(f"Error playing audio: {e}")


def get_tts_engine(engine_type=TTS_ENGINE):
    """Factory function to get the appropriate TTS engine"""
    if engine_type.lower() == "gtts":
        return GTTSEngine()
    elif engine_type.lower() == "pyttsx3":
        return PyttsxEngine()
    elif engine_type.lower() == "openai":
        return OpenAITTSEngine()
    else:
        raise ValueError(f"Unsupported TTS engine: {engine_type}")
