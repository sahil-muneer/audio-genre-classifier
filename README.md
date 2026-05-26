# 🎵 Full-Stack AI Audio Genre Classifier & Live Analytics

An automated, end-to-end machine learning pipeline and web application that extracts frequency-domain features from raw audio signals to classify tracks into 10 musical genres. Originally developed as an academic Signals & Systems (CCA 2) project, this system has been scaled into a full-stack, cloud-connected architecture featuring live NoSQL database logging and dynamic visual analytics.

## ✨ Key Features & Architecture
* **AI Machine Learning Pipeline:** Utilizes a trained Random Forest classification model to analyze raw acoustic data and predict genres with high accuracy.
* **Acoustic Signal Processing:** Extracts 13 Mel-Frequency Cepstral Coefficients (MFCCs) to mathematically map human hearing perception and discard irrelevant acoustic noise.
* **Real-Time Frequency Visualization:** Automatically generates and renders a Mel-Spectrogram heat-map of the uploaded audio signal on the frontend using `matplotlib` and `librosa`.
* **Live Cloud Analytics Dashboard:** Seamlessly integrated with Firebase Firestore. Every prediction is logged to the cloud in real-time, dynamically updating a custom Chart.js analytics dashboard via a split-screen UI.
* **Modular "Edge-Ready" Architecture:** Built as an isolated Flask microservice. The backend handles core ML logic and database routing independently, ensuring zero-conflict integration for future frontend frameworks or IoT hardware nodes.

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **Machine Learning & Math:** `scikit-learn`, `librosa`, `numpy`, `joblib`
* **Data Visualization:** `matplotlib` (Spectrograms), `Chart.js` (Live Database Analytics)
* **Cloud Database:** Firebase Admin SDK (Firestore NoSQL)
* **Frontend:** HTML5, CSS3 (Flexbox), JavaScript

## 📊 Dataset & Model Training
This project was trained on the standard **GTZAN Dataset**:
* **1,000 Audio Tracks** (30 seconds each, 22,050 Hz).
* **10 Genres:** Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock.
* **Analysis:** The model achieved strong overall accuracy, excelling in Pop (80%+ accuracy), Metal, and Classical. Minor overlap occurred between Rock and Blues due to shared instrumentation and similar MFCC frequency bands. 
* *Note: The raw audio dataset is ignored in this repository due to size constraints, but the extracted mathematical features (`extracted_features.csv`) are provided.*

## 🚀 Local Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd genre_classifier

---
**Developed by:** Sahil Muneer Nowsheri