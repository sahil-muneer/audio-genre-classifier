import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "Data/genres_original"
OUTPUT_CSV = "extracted_features.csv"

def extract_mfcc_features(dataset_path):
    data = []
    
    # Loop through all the genre folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            genre_label = os.path.basename(dirpath)
            print(f"Processing genre: {genre_label}...")
            
            for f in filenames:
                if f.endswith('.wav'):
                    file_path = os.path.join(dirpath, f)
                    
                    try:
                        # Load audio file 
                        signal, sr = librosa.load(file_path, sr=22050, duration=30)
                        
                        # Extract 13 MFCC coefficients
                        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
                        
                        # Take the average of the MFCCs over time
                        mfccs_averaged = np.mean(mfccs.T, axis=0)
                        
                        row = list(mfccs_averaged)
                        row.append(genre_label)
                        data.append(row)
                        
                    except Exception as e:
                        # Handles any corrupted files smoothly
                        print(f"Skipping {f} due to error: {e}")

    columns = [f"mfcc_{idx}" for idx in range(13)] + ["genre"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccess! Features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_mfcc_features(DATASET_PATH)