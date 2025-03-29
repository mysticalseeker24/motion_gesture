# Frontend Integration Guide

## Quick Start

This guide provides step-by-step instructions to connect the gesture recognition backend to a React Native frontend application using Expo CLI.

## System Architecture (Detailed)

```
┌─────────────────────────────────────────────┐           ┌─────────────────────────────────────────────┐
│            BACKEND SYSTEM                    │           │            FRONTEND SYSTEM                   │
│ ┌─────────────────────┐ ┌─────────────────┐ │           │ ┌─────────────────────┐ ┌─────────────────┐ │
│ │ PyTorch Local Model │ │ OpenAI Vision   │ │           │ │ Expo Camera Module  │ │ UI Components   │ │
│ │ - Fast Detection    │ │ - Enhanced      │ │           │ │ - Frame Capture     │ │ - Display       │ │
│ │ - Gesture Classes   │ │   Analysis      │ │           │ │ - Quality Control   │ │ - User Input    │ │
│ └─────────┬───────────┘ └────────┬────────┘ │           │ └─────────┬───────────┘ └────────┬────────┘ │
│           │                      │          │           │           │                      │          │
│ ┌─────────▼───────────────────────▼────────┐ │   HTTP   │ ┌─────────▼───────────────────────▼────────┐ │
│ │                                           │ │   REST   │ │                                           │ │
│ │           FastAPI Endpoints              │◄────────────►│          Axios API Client                │ │
│ │                                           │ │   JSON   │ │                                           │ │
│ └─────────────────────┬─────────────────────┘ │           │ └─────────────────────┬─────────────────────┘ │
│                       │                       │           │                       │                       │
│ ┌─────────────────────▼─────────────────────┐ │           │ ┌─────────────────────▼─────────────────────┐ │
│ │                                           │ │           │ │                                           │ │
│ │           TTS Audio Generation            │ │           │ │           Audio Playback                  │ │
│ │           - Google TTS                    │ │           │ │           - Expo AV                       │ │
│ │           - OpenAI TTS                    │ │           │ │           - React Native Sound            │ │
│ │           - pyttsx3                       │ │           │ │                                           │ │
│ └───────────────────────────────────────────┘ │           │ └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘           └─────────────────────────────────────────────┘
         │                            ▲                               │                            ▲
         │                            │                               │                            │
         ▼                            │                               ▼                            │
┌─────────────────────────────────────────────┐           ┌─────────────────────────────────────────────┐
│                                             │           │                                             │
│  OpenAI API Services                        │           │  Device Hardware                            │
│  - GPT-4o Vision API                        │           │  - Camera                                   │
│  - Text-to-Speech API                       │           │  - Speakers                                 │
│  - Rate Limiting & Caching                  │           │  - Touch Interface                          │
│                                             │           │                                             │
└─────────────────────────────────────────────┘           └─────────────────────────────────────────────┘
```

## Required Frontend Dependencies

Add these to your Expo project:

```bash
npx expo install expo-camera expo-av expo-file-system
npx expo install @react-native-async-storage/async-storage
npm install axios
```

## Integration Steps

### 1. Setup Optimized API Connection

Create an efficient API utility using axios with request cancellation for better performance:

```javascript
// src/utils/api.js
import axios from 'axios';

// Configure base API settings
const API_BASE_URL = 'http://your-backend-ip:8000';

// Create an axios instance with optimized settings
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 8000, // Reduced timeout for faster error detection
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
});

// Track request cancellation tokens
const cancelTokens = {};

// Cancel any pending requests with the given ID
const cancelPendingRequests = (requestId) => {
  if (cancelTokens[requestId]) {
    cancelTokens[requestId].cancel('Operation canceled due to new request.');
    delete cancelTokens[requestId];
  }
};

// Get all available gestures (with caching)
let cachedGestures = null;
export const getGestures = async () => {
  try {
    // Return cached result if available
    if (cachedGestures) return cachedGestures;
    
    const response = await api.get('/gestures');
    cachedGestures = response.data.gestures;
    return cachedGestures;
  } catch (error) {
    console.error('Error fetching gestures:', error);
    return [];
  }
};

// Detect gesture from base64 image
export const detectGesture = async (base64Image, requestId = 'detect') => {
  // Cancel any pending requests with same ID
  cancelPendingRequests(requestId);
  
  // Create a new cancel token
  const source = axios.CancelToken.source();
  cancelTokens[requestId] = source;
  
  try {
    // Optimize by sending minimal data
    // Remove 'data:image/jpeg;base64,' prefix if present
    const imageData = base64Image.includes('data:image') 
      ? base64Image.split(',')[1] 
      : base64Image;
    
    const response = await api.post('/detect', {
      image: imageData
    }, {
      cancelToken: source.token
    });
    
    // Clean up the cancel token
    delete cancelTokens[requestId];
    return response.data;
  } catch (error) {
    if (axios.isCancel(error)) {
      console.log('Request canceled:', error.message);
      return null;
    }
    console.error('Error detecting gesture:', error);
    return null;
  }
};

// Generate speech from text
export const textToSpeech = async (text, requestId = 'tts') => {
  cancelPendingRequests(requestId);
  const source = axios.CancelToken.source();
  cancelTokens[requestId] = source;
  
  try {
    const response = await api.post('/text-to-speech', {
      text: text
    }, {
      responseType: 'blob',
      cancelToken: source.token
    });
    
    delete cancelTokens[requestId];
    return response.data;
  } catch (error) {
    if (axios.isCancel(error)) {
      console.log('Request canceled:', error.message);
      return null;
    }
    console.error('Error generating speech:', error);
    return null;
  }
};

// Clean up function to call when component unmounts
export const cleanupAPI = () => {
  Object.keys(cancelTokens).forEach(id => {
    cancelTokens[id].cancel('Request canceled due to component unmount');
  });
};

export default api;
```

### 2. Create Optimized Camera Component (Based on Expo Documentation)

```jsx
// src/components/GestureCamera.js
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { View, StyleSheet, Platform } from 'react-native';
import { Camera } from 'expo-camera';

// Optimized camera component based on Expo documentation
const GestureCamera = ({ onFrame, isActive }) => {
  const [hasPermission, setHasPermission] = useState(null);
  const cameraRef = useRef(null);
  const lastProcessedTime = useRef(0);
  const processingFrame = useRef(false);
  const frameInterval = 300; // ms between frames (adjust based on performance needs)
  
  // Request camera permissions only once on mount
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);
  
  // Optimal camera settings for performance
  const cameraSettings = useMemo(() => ({
    // Use lower quality for faster processing
    quality: Platform.OS === 'ios' ? 0.5 : 0.4, 
    base64: true,
    exif: false, // Don't need EXIF data
    skipProcessing: true, // Skip extra image processing
    // Use a smaller image if on Android for speed
    width: Platform.OS === 'android' ? 640 : undefined,
    height: Platform.OS === 'android' ? 480 : undefined,
    fixOrientation: false, // Handle orientation in backend if needed
    imageType: 'jpeg', // JPEG is faster than PNG
  }), []);
  
  // Frame capturing logic - runs continuously when active
  const captureFrame = useCallback(async () => {
    if (!isActive || !cameraRef.current || processingFrame.current) return;
    
    const now = Date.now();
    if (now - lastProcessedTime.current < frameInterval) return;
    
    try {
      processingFrame.current = true;
      const photo = await cameraRef.current.takePictureAsync(cameraSettings);
      lastProcessedTime.current = now;
      processingFrame.current = false;
      
      // Call parent handler with the captured frame
      if (onFrame && photo.base64) {
        onFrame(photo.base64);
      }
    } catch (error) {
      console.error('Error capturing frame:', error);
      processingFrame.current = false;
    }
  }, [isActive, onFrame, cameraSettings]);
  
  // Set up frame capture interval
  useEffect(() => {
    let frameId;
    
    const scheduleNextFrame = () => {
      frameId = requestAnimationFrame(() => {
        captureFrame().finally(() => {
          if (isActive) scheduleNextFrame();
        });
      });
    };
    
    if (isActive && hasPermission) {
      scheduleNextFrame();
    }
    
    return () => {
      if (frameId) cancelAnimationFrame(frameId);
    };
  }, [isActive, hasPermission, captureFrame]);
  
  if (hasPermission === null) {
    return <View style={styles.container} />;
  }
  
  if (hasPermission === false) {
    return <View style={styles.container} />;
  }
  
  return (
    <Camera
      ref={cameraRef}
      style={styles.camera}
      type={Camera.Constants.Type.back}
      ratio="16:9"
      // Use lower resolution when possible for better performance
      videoStabilizationMode={Camera.Constants.VideoStabilization.off}
      // Additional optimizations based on Expo Camera docs
      useCamera2Api={Platform.OS === 'android'} // Better performance on Android
      pictureSize="640x480" // Smaller picture size for faster processing
    />
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
});

export default React.memo(GestureCamera); // Use memo for preventing unnecessary re-renders
```

### 3. Create Optimized Audio Playback Utility

```javascript
// src/utils/audioPlayer.js
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';

// Audio cache for previously played sounds
const audioCache = {};
let currentSound = null;

// Initialize audio settings once
export const initAudio = async () => {
  try {
    await Audio.setAudioModeAsync({
      playsInSilentModeIOS: true,
      staysActiveInBackground: false,
      shouldDuckAndroid: true,
    });
    return true;
  } catch (error) {
    console.error('Error initializing audio:', error);
    return false;
  }
};

// Convert blob to file and play
export const playAudioFromBlob = async (audioBlob, cacheKey = null) => {
  try {
    // Check cache first if cacheKey provided
    if (cacheKey && audioCache[cacheKey]) {
      await playSound(audioCache[cacheKey]);
      return true;
    }
    
    // Create unique temporary filename
    const fileName = `temp_audio_${Date.now()}.mp3`;
    const fileUri = `${FileSystem.cacheDirectory}${fileName}`;
    
    // Convert blob to base64 and save to file
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = async () => {
        try {
          const base64data = reader.result.split(',')[1];
          await FileSystem.writeAsStringAsync(fileUri, base64data, {
            encoding: FileSystem.EncodingType.Base64,
          });
          
          // Load and play the sound
          const { sound } = await Audio.Sound.createAsync(
            { uri: fileUri }, 
            { shouldPlay: true, volume: 1.0 }
          );
          
          // Cache if needed
          if (cacheKey) {
            audioCache[cacheKey] = sound;
          }
          
          // Keep reference to current sound
          if (currentSound) {
            await currentSound.unloadAsync();
          }
          currentSound = sound;
          
          // Add completion handler
          sound.setOnPlaybackStatusUpdate(status => {
            if (status.didJustFinish) {
              if (!cacheKey) {
                // Unload uncached sounds when finished
                sound.unloadAsync();
                // Clean up temp file
                FileSystem.deleteAsync(fileUri, { idempotent: true });
              }
            }
          });
          
          resolve(true);
        } catch (error) {
          console.error('Error processing audio:', error);
          reject(error);
        }
      };
      reader.onerror = () => reject(new Error('FileReader error'));
      reader.readAsDataURL(audioBlob);
    });
  } catch (error) {
    console.error('Error playing audio from blob:', error);
    return false;
  }
};

// Play a pre-loaded sound
const playSound = async (sound) => {
  try {
    if (currentSound && currentSound !== sound) {
      await currentSound.stopAsync().catch(() => {});
    }
    currentSound = sound;
    await sound.setPositionAsync(0);
    await sound.playAsync();
    return true;
  } catch (error) {
    console.error('Error playing sound:', error);
    return false;
  }
};

// Clean up all sounds
export const cleanupAudio = async () => {
  try {
    if (currentSound) {
      await currentSound.unloadAsync();
      currentSound = null;
    }
    
    // Unload all cached sounds
    const cacheSounds = Object.values(audioCache);
    await Promise.all(cacheSounds.map(sound => sound.unloadAsync()));
    
    // Clear cache
    Object.keys(audioCache).forEach(key => delete audioCache[key]);
    
    return true;
  } catch (error) {
    console.error('Error cleaning up audio:', error);
    return false;
  }
};
```

### 4. Main Optimized Gesture Recognition Screen

```jsx
// src/screens/GestureRecognitionScreen.js
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, StatusBar } from 'react-native';
import { detectGesture, cleanupAPI } from '../utils/api';
import { initAudio, playAudioFromBlob, cleanupAudio } from '../utils/audioPlayer';
import GestureCamera from '../components/GestureCamera';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Constants
const DETECTION_COOLDOWN = 1000; // ms between detections
const CONFIDENCE_THRESHOLD = 0.65; // Minimum confidence to accept

const GestureRecognitionScreen = () => {
  // State management with minimal re-renders
  const [isActive, setIsActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastDetectionTime, setLastDetectionTime] = useState(0);
  const [detectedGesture, setDetectedGesture] = useState(null);
  
  // Initialize audio on mount
  useEffect(() => {
    initAudio();
    
    // Cleanup on unmount
    return () => {
      cleanupAPI();
      cleanupAudio();
    };
  }, []);
  
  // Load saved settings
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const savedSettings = await AsyncStorage.getItem('gestureSettings');
        if (savedSettings) {
          // Apply saved settings if needed
          console.log('Loaded settings:', JSON.parse(savedSettings));
        }
      } catch (error) {
        console.error('Error loading settings:', error);
      }
    };
    
    loadSettings();
  }, []);
  
  // Process each captured frame
  const handleFrame = useCallback(async (base64Image) => {
    // Skip if already processing or in cooldown period
    const now = Date.now();
    if (isProcessing || now - lastDetectionTime < DETECTION_COOLDOWN) return;
    
    try {
      setIsProcessing(true);
      
      // Detect gesture using API
      const result = await detectGesture(base64Image);
      
      // Process valid detection
      if (result && result.gesture && result.confidence > CONFIDENCE_THRESHOLD) {
        setDetectedGesture(result);
        setLastDetectionTime(now);
        
        // Generate and play speech
        if (result.interpretation) {
          const response = await fetch(`http://your-backend-ip:8000/text-to-speech`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: result.interpretation }),
          });
          
          if (response.ok) {
            const audioBlob = await response.blob();
            await playAudioFromBlob(audioBlob, result.gesture); // Cache common gestures
          }
        }
      }
    } catch (error) {
      console.error('Error processing frame:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, lastDetectionTime]);
  
  // Memoize UI components to minimize re-renders
  const StatusDisplay = useMemo(() => (
    <View style={styles.statusBar}>
      <Text style={styles.statusText}>
        {isActive ? 'Analyzing Gestures' : 'Paused'}
      </Text>
      {isProcessing && <ActivityIndicator color="#fff" size="small" />}
    </View>
  ), [isActive, isProcessing]);
  
  const DetectionDisplay = useMemo(() => {
    if (!detectedGesture) return null;
    
    return (
      <View style={styles.detectionContainer}>
        <Text style={styles.gestureLabel}>
          {detectedGesture.gesture} ({Math.round(detectedGesture.confidence * 100)}%)
        </Text>
        {detectedGesture.interpretation && (
          <Text style={styles.interpretation}>{detectedGesture.interpretation}</Text>
        )}
      </View>
    );
  }, [detectedGesture]);
  
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      
      {/* Camera component (only active when needed) */}
      <GestureCamera 
        onFrame={handleFrame}
        isActive={isActive}
      />
      
      {/* UI Overlay */}
      <View style={styles.overlay}>
        {StatusDisplay}
        {DetectionDisplay}
        
        <TouchableOpacity
          style={[styles.button, isActive ? styles.stopButton : styles.startButton]}
          onPress={() => setIsActive(!isActive)}
          activeOpacity={0.7}
        >
          <Text style={styles.buttonText}>
            {isActive ? 'Stop' : 'Start'} Recognition
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    paddingTop: 40,
    paddingBottom: 20,
    paddingHorizontal: 20,
    flexDirection: 'column',
    justifyContent: 'space-between',
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 8,
    borderRadius: 4,
  },
  statusText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  detectionContainer: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 16,
    borderRadius: 8,
    marginTop: 'auto',
    marginBottom: 20,
  },
  gestureLabel: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  interpretation: {
    color: '#fff',
    fontSize: 16,
  },
  button: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  startButton: {
    backgroundColor: '#4CAF50',
  },
  stopButton: {
    backgroundColor: '#F44336',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 18,
  },
});

export default GestureRecognitionScreen;
```

### 5. App Entry Point with Performance Optimizations

```jsx
// App.js
import React, { useEffect } from 'react';
import { SafeAreaView, StyleSheet, LogBox, Platform, AppState } from 'react-native';
import GestureRecognitionScreen from './src/screens/GestureRecognitionScreen';

// Ignore specific warnings that don't affect functionality
LogBox.ignoreLogs([
  'Require cycle:', // Ignore require cycles
  'ViewPropTypes will be removed', // Ignore deprecation warnings
]);

// Main app with optimized lifecycle management
export default function App() {
  // Handle app state changes to optimize resource usage
  useEffect(() => {
    const subscription = AppState.addEventListener('change', nextAppState => {
      if (nextAppState.match(/inactive|background/)) {
        // App going to background - release resources
        console.log('App entering background - releasing resources');
      } else if (nextAppState === 'active') {
        // App coming to foreground - reinitialize if needed
        console.log('App entering foreground - reinitializing');
      }
    });

    return () => {
      subscription.remove();
    };
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <GestureRecognitionScreen />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
});
```

## Optimization Best Practices for Expo-based Applications

### Camera Optimization

Based on Expo Camera documentation, implemented the following optimizations:

1. **Reduced Resolution**: Using smaller image sizes (640x480) for faster processing
2. **Lower Quality Settings**: Setting JPEG quality to 0.4-0.5 for reduced data size
3. **Skip Processing**: Using `skipProcessing: true` to avoid additional image processing
4. **Camera2 API**: Using `useCamera2Api` on Android for better performance
5. **Frame Rate Control**: Using `captureFrame` with intelligent throttling instead of continuous capture

### React Performance Optimizations

1. **Memoization**: Using `useMemo` and `useCallback` to prevent unnecessary re-renders
2. **Component Splitting**: Separating camera logic from UI for better performance
3. **Refs Instead of State**: Using refs for values that don't need to trigger re-renders
4. **Request Animation Frame**: Using rAF for smoother frame capture scheduling

### Network Optimizations

1. **Request Cancellation**: Implementing axios cancellation tokens to prevent stale responses
2. **Data Minimization**: Sending only necessary data to the backend
3. **Caching**: Implementing client-side caching for common results
4. **Audio Caching**: Storing commonly used audio responses for quick playback

## Testing the Integration

1. Start the backend server:
   ```bash
   cd backend
   python main.py
   ```

2. Note the IP address of your backend server (ensure it's accessible from your device)

3. Update the API URLs in the code with your backend IP

4. Start the Expo app:
   ```bash
   npx expo start
   ```

5. Connect using the Expo Go app on your device

## Troubleshooting

### Performance Issues

If the app is running slowly:

1. **Adjust Frame Rate**: Increase `DETECTION_COOLDOWN` to reduce processing frequency
2. **Lower Resolution**: Decrease camera resolution further
3. **Check Network**: Ensure good connectivity between device and backend
4. **Debugging**: Use Expo DevTools to monitor performance metrics

```javascript
// Add this import to monitor performance
import { PerformanceObserver, performance } from 'perf_hooks';

// Then add measurement code around critical sections
performance.mark('frameProcessStart');
// ... processing code ...
performance.mark('frameProcessEnd');
performance.measure('Frame Processing', 'frameProcessStart', 'frameProcessEnd');
```

### Camera Issues

Common issues based on Expo documentation:

1. **Android Permissions**: Some devices require additional permissions handling
2. **iOS Simulator**: Camera doesn't work in iOS simulators, use physical device
3. **Expo Go Limitations**: Some camera features work better in standalone builds

## Production Deployment

For optimal performance in production:

1. Create a standalone app with EAS Build:
   ```bash
   npx eas build --platform all
   ```

2. Enable Hermes JavaScript engine for better performance

3. Consider using a dedicated server for the backend with appropriate scaling

---
