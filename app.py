from flask import Flask, request, render_template
import librosa
import numpy as np
import joblib
import os
import webbrowser
from threading import Timer
from datetime import datetime
import time

# --- NEW GRAPHING IMPORTS ---
import matplotlib
matplotlib.use('Agg') # This stops the graph from popping up in a separate window
import matplotlib.pyplot as plt
import librosa.display

# Firebase Imports
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# --- FIREBASE SETUP ---
print("Connecting to Cloud Database...")
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase connected successfully!")
except Exception as e:
    print(f"Firebase setup skipped or failed: {e}")
    db = None

# --- LOAD AI MODEL ---
print("Loading AI Model...")
model = joblib.load("saved_model.pkl")
encoder = joblib.load("saved_encoder.pkl")

def get_dashboard_data():
    if not db: return {}
    try:
        docs = db.collection('predictions').stream()
        genre_counts = {}
        for doc in docs:
            genre = doc.to_dict().get('predicted_genre', 'UNKNOWN')
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        return genre_counts
    except: return {}

@app.route("/", methods=["GET"])
def index():
    counts = get_dashboard_data()
    return render_template("index.html", prediction=None, data=counts)

@app.route("/predict", methods=["POST"])
def predict():
    if "audio_file" not in request.files:
        return "No file uploaded", 400
        
    file = request.files["audio_file"]
    if file.filename == "":
        return "No file selected", 400

    filepath = "temp_upload.wav"
    file.save(filepath)

    try:
        # 1. Signal Processing
        signal, sr = librosa.load(filepath, sr=22050, duration=30)
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
        
        # --- NEW: GENERATE SPECTROGRAM GRAPH ---
        S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        plt.figure(figsize=(5, 2))
        librosa.display.specshow(S_dB, sr=sr, cmap='magma')
        plt.tight_layout(pad=0)
        
        # Create a static folder if it doesn't exist to hold the image
        os.makedirs("static", exist_ok=True)
        img_path = os.path.join("static", "spectrogram.png")
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        # ---------------------------------------

        # 2. Prediction
        features = mfccs.reshape(1, -1)
        prediction_encoded = model.predict(features)
        prediction_text = encoder.inverse_transform(prediction_encoded)[0].upper()
        
        # 3. Log to Firebase
        if db:
            db.collection('predictions').document().set({
                'filename': file.filename,
                'predicted_genre': prediction_text,
                'timestamp': datetime.now()
            })
            print(f"Logged to cloud: {file.filename} -> {prediction_text}")
        
        os.remove(filepath) 
        
        updated_counts = get_dashboard_data()
        
        # We pass a timestamp to force the browser to update the image
        current_time = int(time.time())
        return render_template("index.html", prediction=prediction_text, data=updated_counts, timestamp=current_time)
        
    except Exception as e:
        return f"Error processing audio: {e}", 500

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5001")

if __name__ == "__main__":
    Timer(1.5, open_browser).start()
    app.run(port=5001, debug=True, use_reloader=False)