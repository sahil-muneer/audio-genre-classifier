import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# 1. Load the extracted features
print("Loading data...")
df = pd.read_csv("extracted_features.csv")
print(f"Number of columns loaded: {len(df.columns)}")

# 2. Separate features (X) and the answers/labels (y)
X = df.drop("genre", axis=1) # All the math stuff
y = df["genre"]              # The actual genre names

# 3. Convert text labels (blues, rock) to numbers for the ML model
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 4. Split data: 80% for training the AI, 20% for testing it
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Initialize and train the Random Forest Classifier
print("Training the AI model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions on the 20% test data
y_pred = model.predict(X_test)

# 7. Evaluate the results
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Complete!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# 8. Plot the Confusion Matrix (Screenshot this for your presentation!)
print("Generating Confusion Matrix plot...")
disp = ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, 
    display_labels=encoder.classes_, 
    cmap=plt.cm.Blues,
    xticks_rotation='vertical'
)
plt.title("Genre Classification Confusion Matrix")
plt.tight_layout()
plt.show()

import joblib

# Save the trained model and the label encoder
print("Saving model for web app...")
joblib.dump(model, "saved_model.pkl")
joblib.dump(encoder, "saved_encoder.pkl")
print("Model saved successfully!")