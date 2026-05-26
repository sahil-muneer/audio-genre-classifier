import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "Data/genres_original"
OUTPUT_CSV = "extracted_features_pro.csv" # New file name so we don't overwrite the old one!

def extract_advanced_features(dataset_path):
    data = []
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            genre_label = os.path.basename(dirpath)
            print(f"Extracting advanced features for: {genre_label}...")
            
            for f in filenames:
                if f.endswith('.wav'):
                    file_path = os.path.join(dirpath, f)
                    
                    try:
                        # Load audio file
                        signal, sr = librosa.load(file_path, sr=22050, duration=30)
                        
                        # 1. MFCCs (13 features) - Timbre/Texture
                        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
                        
                        # 2. Chroma (12 features) - Harmony/Pitch classes
                        chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr).T, axis=0)
                        
                        # 3. Spectral Centroid (1 feature) - Brightness of sound
                        centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr).T, axis=0)
                        
                        # 4. Zero-Crossing Rate (1 feature) - Amount of noise/distortion
                        zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal).T, axis=0)
                        
                        # Combine all features into one giant array (27 numbers total)
                        features = np.hstack((mfccs, chroma, centroid, zcr))
                        
                        row = list(features)
                        row.append(genre_label)
                        data.append(row)
                        
                    except Exception as e:
                        print(f"Skipping {f} due to error: {e}")

    # Generate column names dynamically
    columns = [f"mfcc_{idx}" for idx in range(13)] + \
              [f"chroma_{idx}" for idx in range(12)] + \
              ["centroid", "zcr", "genre"]
              
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccess! Advanced features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_advanced_features(DATASET_PATH)