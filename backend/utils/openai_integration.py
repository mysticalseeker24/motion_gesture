import os
import json
import base64
import time
from typing import Optional, Dict, List, Any, Union
import requests
from io import BytesIO
from PIL import Image
import openai
from openai import OpenAI
from config.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TTS_MODEL, TEMPERATURE, MAX_TOKENS

class OpenAIAssistant:
    """Class to interact with OpenAI API for gesture interpretation and response generation"""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        self.model = OPENAI_MODEL
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.tts_model = OPENAI_TTS_MODEL
        print("OpenAI client initialized successfully")
    
    def enhance_interpretation(self, gesture_label: str, image_data=None, context=None) -> Dict[str, Any]:
        """Use OpenAI to enhance a basic gesture label with rich interpretation
        
        Args:
            gesture_label: The basic gesture label detected by the CV model
            image_data: Optional binary image data to send to OpenAI
            context: Optional contextual information about the user/environment
            
        Returns:
            Dictionary with enhanced interpretation and potential actions
        """
        try:
            system_prompt = (
                "You are an AI assistant specializing in interpreting human hand gestures and translating them into rich, "
                "meaningful expressions. Based on the hand gesture and any contextual information provided, generate a "
                "detailed interpretation of what the person might be communicating."
            )
            
            user_prompt = f"Hand Gesture: {gesture_label}\n"
            
            if context:
                user_prompt += f"Context: {context}\n"
            
            user_prompt += (
                "Please provide a detailed interpretation of this gesture, including:\n"
                "1. The likely meaning and intent\n"
                "2. Cultural variations of this gesture if relevant\n"
                "3. Potential follow-up actions or responses\n"
                "4. Any additional meaning that might be conveyed through this gesture"
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # If we have image data, add it to the messages
            if image_data is not None:
                encoded_image = self._encode_image(image_data)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]}
                ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            interpretation = response.choices[0].message.content
            
            return {
                "original_gesture": gesture_label,
                "enhanced_interpretation": interpretation,
                "confidence": response.choices[0].finish_reason == "stop",
                "model": self.model
            }
            
        except Exception as e:
            print(f"Error with OpenAI interpretation: {e}")
            return {
                "original_gesture": gesture_label,
                "enhanced_interpretation": f"This appears to be a {gesture_label} gesture.",
                "error": str(e),
                "confidence": 0.5
            }
    
    def analyze_gesture_from_image(self, image_data):
        """Directly analyze a hand gesture from image data using OpenAI vision capabilities
        
        Args:
            image_data: Binary image data containing the hand gesture
            
        Returns:
            Dictionary with gesture analysis and interpretation
        """
        try:
            encoded_image = self._encode_image(image_data)
            
            system_prompt = (
                "You are an AI assistant specializing in analyzing and interpreting hand gestures from images. "
                "Please look at the provided image and:"                    
                "1. Identify if there is a hand gesture present"               
                "2. Describe what the gesture is (e.g., waving, pointing, thumbs up, sign language, etc.)"                  
                "3. Explain the common meaning or intent of this gesture"                 
                "4. If applicable, identify if this is a sign language gesture and provide its meaning"                     
                "Your analysis should be concise but thorough. If no clear hand gesture is visible, just state that fact."
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Please analyze this hand gesture and tell me what it might mean:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]}
                ],
                temperature=0.2,  # Lower temperature for more factual responses
                max_tokens=self.max_tokens
            )
            
            analysis = response.choices[0].message.content
            
            # Extract a simple gesture label from the detailed analysis
            gesture_label = self._extract_gesture_label(analysis)
            
            return {
                "gesture_label": gesture_label,
                "detailed_analysis": analysis,
                "confidence": response.choices[0].finish_reason == "stop",
                "model": self.model
            }
            
        except Exception as e:
            print(f"Error with OpenAI image analysis: {e}")
            return {
                "gesture_label": "unknown",
                "detailed_analysis": "Unable to analyze the image.",
                "error": str(e),
                "confidence": 0
            }
    
    def analyze_gesture_image(self, image_path: str, prompt: str = None) -> str:
        """Analyze an image containing a gesture using OpenAI Vision capabilities
        
        Args:
            image_path: Path to the image file
            prompt: Optional prompt to guide the analysis
            
        Returns:
            String containing the analysis of the gesture
        """
        try:
            if not prompt:
                prompt = "What hand gesture is shown in this image? Please provide a detailed description of the meaning and context of this gesture."
            
            # Read image file
            with open(image_path, "rb") as image_file:
                # Call OpenAI Vision API with GPT-4 Vision model (using current model name)
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Updated from gpt-4-vision-preview to gpt-4o
                    messages=[
                        {"role": "system", "content": "You are an AI assistant specializing in interpreting human hand gestures and translating them into rich, meaningful expressions."},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"},
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
            
            # Extract the response text
            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return "No detailed analysis available."
                
        except Exception as e:
            print(f"Error analyzing gesture image: {e}")
            return f"Error analyzing gesture: {str(e)}"
    
    def generate_text_to_speech(self, text, output_file):
        """Generate speech from text using OpenAI's TTS API
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file
            
        Returns:
            Path to the generated audio file or None if failed
        """
        try:
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice="alloy",  # You can change this to other available voices
                input=text
            )
            
            # Save to file
            response.stream_to_file(output_file)
            return output_file
            
        except Exception as e:
            print(f"Error generating TTS with OpenAI: {e}")
            return None
    
    def generate_response_with_context(self, gesture_label, user_info=None, chat_history=None):
        """Generate a complete response taking into account user context and chat history
        
        Args:
            gesture_label: The detected gesture label
            user_info: Optional dictionary with user information
            chat_history: Optional list of previous chat messages
            
        Returns:
            Full response object with text and action recommendations
        """
        try:
            system_prompt = (
                "You are an AI assistant for a hand gesture recognition application. Your task is to generate "
                "helpful, informative responses to users based on their hand gestures and any available context. "
                "Your responses should be natural, compassionate, and tailored to the user's likely needs."
            )
            
            user_prompt = f"Detected hand gesture: {gesture_label}\n"
            
            if user_info:
                user_prompt += f"User information: {json.dumps(user_info)}\n"
            
            if chat_history:
                user_prompt += f"Previous conversation: {json.dumps(chat_history)}\n"
            
            user_prompt += (
                "Please provide a complete response that includes:\n"
                "1. A natural language interpretation of the gesture\n"
                "2. Recommended actions or follow-ups\n"
                "3. Any relevant information or assistance based on the gesture"
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "text": response.choices[0].message.content,
                "gesture": gesture_label,
                "model": self.model,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"Error generating response with OpenAI: {e}")
            return {
                "text": f"I understood your {gesture_label} gesture. How can I help you?",
                "gesture": gesture_label,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _encode_image(self, image_data):
        """Encode image data to base64 for API transmission"""
        if isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode('utf-8')
        else:
            # Convert numpy array or other image format to bytes
            try:
                img = Image.fromarray(image_data)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Error encoding image: {e}")
                return None
    
    def _extract_gesture_label(self, analysis):
        """Extract a simple gesture label from a detailed analysis"""
        # This is a simple extraction that could be made more sophisticated
        # Currently just looks for key gesture terms in the analysis
        analysis = analysis.lower()
        
        # Check for common gestures
        gesture_keywords = {
            "wave": "hello",
            "waving": "hello",
            "greeting": "hello",
            "help": "helpme",
            "assistance": "helpme",
            "love": "iloveyou",
            "heart": "iloveyou",
            "namaste": "namaste",
            "prayer": "namaste",
            "no": "no",
            "disagree": "no",
            "negative": "no",
            "hospital": "searching hospital",
            "medical": "searching hospital",
            "police": "searching police",
            "emergency": "searching police",
            "thank": "thanks",
            "grateful": "thanks",
            "yes": "yes",
            "agree": "yes",
            "affirmative": "yes"
        }
        
        for keyword, label in gesture_keywords.items():
            if keyword in analysis:
                return label
        
        # If no specific keyword is found, return a general label
        return "unknown_gesture"
