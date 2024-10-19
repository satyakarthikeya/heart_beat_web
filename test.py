import numpy as np
import librosa
import pywt
from model_predictor import ModelPredictor

# Function to extract features from a WAV file
def extract_features(file_path):
    features = []
    y, sr = librosa.load(file_path, sr=None)

    if len(y) == 0:
        print(f"Warning: The audio file {file_path} is empty or not valid.")
        return np.array([])  # Return an empty array if the audio is empty

    print(f"Loaded audio file: {file_path} with sample rate: {sr}")

    # Log Mel Spectrogram
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spect = librosa.power_to_db(mel_spect)
    features.append(np.mean(log_mel_spect, axis=1))

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.append(np.mean(mfccs, axis=1))

    # Delta and Delta-Delta (MFCC Derivatives)
    delta_mfcc = librosa.feature.delta(mfccs)
    delta_delta_mfcc = librosa.feature.delta(mfccs, order=2)
    features.append(np.mean(delta_mfcc, axis=1))
    features.append(np.mean(delta_delta_mfcc, axis=1))

    # Discrete Wavelet Transform (DWT)
    coeffs = pywt.wavedec(y, 'db1', level=5)
    features.append(np.hstack([np.mean(c) for c in coeffs]))

    # Chroma Feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma, axis=1))

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spectral_centroid))

    # Spectral Contrast with adjusted fmin
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=20.0)
    features.append(np.mean(spectral_contrast, axis=1))

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))

    # Temporal Flatness (Spectral Flatness)
    flatness = librosa.feature.spectral_flatness(y=y)
    features.append(np.mean(flatness))

    print(f"Extracted features from {file_path}: {features}")

    return np.hstack(features)

# Load the model
model_path = 'xgboost_heart_classification_model.joblib'
model_predictor = ModelPredictor(model_path)

# Path to your WAV file
wav_file_path = r"C:\Users\pasar\Documents\Sound Recordings\king_3.wav"  # Update this path to your actual WAV file location

# Extract features from the WAV file
extracted_features = extract_features(wav_file_path)

# Reshape features for the model
extracted_features = extracted_features.reshape(1, -1)

# Make a prediction
prediction = model_predictor.predict(extracted_features)

# Print the prediction result
print(f'Prediction: {prediction}')
