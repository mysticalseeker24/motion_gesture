import os
import time
import gtts
import pyttsx3
from pathlib import Path

from config.config import TTS_ENGINE, TTS_LANGUAGE, TTS_OUTPUT_DIR, OPENAI_API_KEY, OPENAI_TTS_VOICE
from utils.audio_manager import audio_manager


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
            # Check audio cache first
            cached_path = audio_manager.get_audio_path(text)
            if cached_path:
                self._play_audio(cached_path)
                return cached_path
                
            # Generate a unique filename based on timestamp
            filename = f"tts_{int(time.time())}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create TTS object and save to file
            tts = gtts.gTTS(text=text, lang=self.language, slow=False)
            tts.save(filepath)
            
            # Save to cache
            audio_manager.save_audio(text, filepath)
            
            # Play the audio file (platform-specific)
            self._play_audio(filepath)
            
            return filepath
        except Exception as e:
            print(f"Error in GTTSEngine.speak: {e}")
            return None
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text and save to specified file"""
        try:
            # Check audio cache first
            cached_path = audio_manager.get_audio_path(text)
            if cached_path and not output_file:
                return cached_path
                
            # If output file is specified, use it, otherwise generate a filename
            if not output_file:
                filename = f"tts_{int(time.time())}.mp3"
                output_file = os.path.join(self.output_dir, filename)
            
            # Create TTS object and save to file
            tts = gtts.gTTS(text=text, lang=self.language, slow=False)
            tts.save(output_file)
            
            # Save to cache if using default output
            if not output_file:
                audio_manager.save_audio(text, output_file)
            
            return output_file
        except Exception as e:
            print(f"Error in GTTSEngine.generate_speech: {e}")
            return None
    
    def _play_audio(self, filepath):
        """Play the audio file using platform-specific methods"""
        # Platform-specific audio playback can be implemented here
        # This is just a placeholder - frontend will handle audio playback
        print(f"[Audio would play: {filepath}]")
        return filepath


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
        
        # Create output directory
        self.output_dir = TTS_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def speak(self, text):
        """Convert text to speech using pyttsx3"""
        try:
            # Check audio cache first
            cached_path = audio_manager.get_audio_path(text)
            if cached_path:
                return cached_path
                
            # Generate a unique filename
            filename = f"tts_{int(time.time())}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            # Generate speech and save to file
            self.generate_speech(text, filepath)
            
            # Save to cache
            audio_manager.save_audio(text, filepath)
            
            return filepath
        except Exception as e:
            print(f"Error in PyttsxEngine.speak: {e}")
            return None
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text and save to specified file"""
        try:
            # Check audio cache first
            cached_path = audio_manager.get_audio_path(text)
            if cached_path and not output_file:
                return cached_path
                
            # If output file is specified, use it, otherwise generate a filename
            if not output_file:
                filename = f"tts_{int(time.time())}.mp3"
                output_file = os.path.join(self.output_dir, filename)
            
            # Save to file
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            
            # Save to cache if using default output
            if not output_file:
                audio_manager.save_audio(text, output_file)
            
            return output_file
        except Exception as e:
            print(f"Error in PyttsxEngine.generate_speech: {e}")
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
            # Check audio cache first
            cached_path = audio_manager.get_audio_path(text)
            if cached_path:
                self._play_audio(cached_path)
                return cached_path
                
            # Generate a unique filename based on timestamp
            filename = f"tts_{int(time.time())}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            # Generate audio file
            self._generate_audio(text, filepath)
            
            # Save to cache
            audio_manager.save_audio(text, filepath)
            
            # Play the audio file
            self._play_audio(filepath)
            
            return filepath
        except Exception as e:
            print(f"Error in OpenAITTSEngine.speak: {e}")
            return None
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text and save to specified file"""
        try:
            # Check audio cache first
            cached_path = audio_manager.get_audio_path(text)
            if cached_path and not output_file:
                return cached_path
                
            # If output file is specified, use it, otherwise generate a filename
            if not output_file:
                filename = f"tts_{int(time.time())}.mp3"
                output_file = os.path.join(self.output_dir, filename)
            
            # Generate audio file
            self._generate_audio(text, output_file)
            
            # Save to cache if using default output
            if not output_file:
                audio_manager.save_audio(text, output_file)
            
            return output_file
        except Exception as e:
            print(f"Error in OpenAITTSEngine.generate_speech: {e}")
            return None
    
    def _generate_audio(self, text, filepath):
        """Generate audio file using OpenAI TTS API"""
        import openai
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text
            )
            response.stream_to_file(filepath)
            return filepath
        except Exception as e:
            print(f"Error generating OpenAI TTS: {e}")
            return None
    
    def _play_audio(self, filepath):
        """Play the audio file using platform-specific methods"""
        # Platform-specific audio playback can be implemented here
        # This is just a placeholder - frontend will handle audio playback
        print(f"[Audio would play: {filepath}]")
        return filepath


# Factory function to get the appropriate TTS engine
def get_tts_engine(engine_type=TTS_ENGINE):
    """Returns the appropriate TTS engine based on configuration"""
    if engine_type.lower() == 'gtts':
        return GTTSEngine()
    elif engine_type.lower() == 'pyttsx3':
        return PyttsxEngine()
    elif engine_type.lower() == 'openai':
        return OpenAITTSEngine()
    else:
        print(f"Unknown TTS engine: {engine_type}, defaulting to gtts")
        return GTTSEngine()
