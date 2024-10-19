import numpy as np
import librosa
import pywt
import logging

class FeatureExtractor:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def extract_features(self, file_path):
        try:
            self.logger.debug(f"Loading audio file: {file_path}")
            features = []

            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            self.logger.debug(f"Audio file loaded with sample rate: {sr}, duration: {len(y)/sr:.2f} seconds")

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

            # Combine all features into a single array
            all_features = np.hstack(features)
            self.logger.debug(f"Extracted features shape: {all_features.shape}")

            return all_features

        except Exception as e:
            logging.error('Error in feature extraction: %s', str(e))
            return None
