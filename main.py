import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input

input_base = "marine_sounds"
output_base = "converted_wavs"
os.makedirs(output_base, exist_ok=True)

for category in os.listdir(input_base):
    input_folder = os.path.join(input_base, category)
    output_folder = os.path.join(output_base, category)
    os.makedirs(output_folder, exist_ok=True)

    if os.path.isdir(input_folder):
        for file in os.listdir(input_folder):
            if file.endswith(".mp3"):
                mp3_path = os.path.join(input_folder, file)
                wav_path = os.path.join(output_folder, file.replace(".mp3", ".wav"))

                try:
                    sound = AudioSegment.from_mp3(mp3_path)
                    sound.export(wav_path, format="wav")
                    print(f"Converted {file} to WAV")
                except Exception as e:
                    print(f"Error converting {file}: {e}")
print(" All MP3 files converted successfully!")

DATASET_PATH = output_base
categories = ["whale", "ship", "background"]
model_path = "marine_model.keras"


def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=5.0)
        if len(audio) == 0:
            raise ValueError("Empty audio")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"⚠ Error extracting features from {file_path}: {e}")
        return None


X, y = [], []
for idx, category in enumerate(categories):
    folder = os.path.join(DATASET_PATH, category)
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder, filename)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(idx)

X = np.array(X)
y = to_categorical(y, num_classes=len(categories))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if os.path.exists(model_path):
    print(" Loading existing model...")
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
else:
    print("🛠️ Training new model...")
    model = Sequential([
        Input(shape=(40,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(categories), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=(X_test, y_test))

    model.save(model_path)
    print(f" Model saved to {model_path}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict(file_path):
    features = extract_features(file_path)
    if features is None:
        print("Could not extract features for prediction.")
        return None
    feat = features.reshape(1, -1)
    prediction = model.predict(feat)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100
    print(f"🎧 Prediction: {categories[class_index]} ({confidence:.2f}% confidence)")
    return categories[class_index]


predict("converted_wavs/whale/Resident Killer Whale .wav")
