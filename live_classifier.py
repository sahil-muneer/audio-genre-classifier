import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import joblib
import os
import time
import warnings
warnings.filterwarnings("ignore")

# 1. Load the trained AI brain
print("Loading AI Brain...")
model = joblib.load("saved_model.pkl")
encoder = joblib.load("saved_encoder.pkl")

# 2. Recording settings
fs = 22050  # Same sample rate we trained on
seconds = 30  # 30-second clips

print("\n🎤 Get ready! Have a song ready to play on your phone...")
time.sleep(3)
print("🔴 RECORDING NOW (Listening for 30 seconds)...")

# 3. Record audio from the laptop microphone
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until the 30 seconds are up
print("✅ Recording complete! Analyzing frequencies...")

# 4. Save to a temporary file so Librosa can read it
temp_file = 'live_temp.wav'
write(temp_file, fs, myrecording)

try:
    # 5. Extract the MFCCs exactly like we did in training
    signal, sr = librosa.load(temp_file, sr=fs, duration=30)
    
    # ✨ THE FIX: Normalize the audio to match studio-level volume ✨
    signal = librosa.util.normalize(signal)
    
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
    
    # 6. Predict the genre
    features = mfccs.reshape(1, -1)
    prediction_encoded = model.predict(features)
    prediction_text = encoder.inverse_transform(prediction_encoded)[0].upper()
    
    print("\n" + "="*40)
    print(f"🎸 PREDICTED GENRE: {prediction_text} 🎸")
    print("="*40 + "\n")

except Exception as e:
    print(f"Error during analysis: {e}")

finally:
    # 7. Clean up the temporary recording file
    #if os.path.exists(temp_file):
    #   os.remove(temp_file)
    pass