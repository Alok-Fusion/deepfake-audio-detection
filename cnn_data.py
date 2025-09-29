# cnn_data.py
"""
Keras Sequence generator that yields log-mel spectrograms for audio files.
Save this as cnn_data.py
"""
import math
import os
import random

import librosa
import numpy as np
from tensorflow.keras.utils import Sequence

# Audio / spectrogram settings (match these with cnn_train and GUI)
SR = 22050
DURATION = 5.0            # seconds (pad/truncate)
N_MELS = 128
FMAX = 8000
N_FFT = 2048
HOP_LENGTH = 512

# Estimate time steps: number of frames for a duration with hop_length
# time_steps = ceil((SR * DURATION - N_FFT) / HOP_LENGTH) + 1
TIME_STEPS = int(np.ceil((SR * DURATION - N_FFT) / HOP_LENGTH)) + 1

def load_audio(path, sr=SR, duration=DURATION):
    """Load an audio file, resample to sr, and pad/truncate to fixed duration."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def mel_spectrogram(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX, time_steps=TIME_STEPS):
    """Compute log-mel spectrogram, normalize to 0..1, pad/truncate time axis to time_steps."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length, fmax=fmax, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    # normalize to 0..1
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    # fix time axis
    if S_norm.shape[1] < time_steps:
        pad_width = time_steps - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
    else:
        S_norm = S_norm[:, :time_steps]
    return S_norm.astype(np.float32)

# Simple augmentations
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return (y + noise_factor * noise).astype(np.float32)

def time_shift(y, shift_max=0.2):
    shift = int(random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

class AudioSequence(Sequence):
    """
    Keras Sequence that yields (X, y) batches:
      X shape: (batch, n_mels, time_steps, 1)
      y shape: (batch,) with values 0.0 or 1.0
    Parameters:
      files: list of file paths
      labels: list/array of 0 (fake) or 1 (real)
      batch_size: int
      shuffle: shuffle at epoch end
      augment: enable simple augmentations on training
    """
    def __init__(self, files, labels, batch_size=32, shuffle=True, augment=False):
        self.files = list(files)
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.files))
        batch_files = self.files[start:end]
        batch_labels = self.labels[start:end]
        X = np.zeros((len(batch_files), N_MELS, TIME_STEPS, 1), dtype=np.float32)
        y = np.zeros((len(batch_files),), dtype=np.float32)
        for i, (fp, lab) in enumerate(zip(batch_files, batch_labels)):
            try:
                wav = load_audio(fp)
                if self.augment:
                    if random.random() < 0.3:
                        wav = add_noise(wav, noise_factor=random.uniform(0.002, 0.01))
                    if random.random() < 0.3:
                        wav = time_shift(wav, shift_max=0.1)
                mel = mel_spectrogram(wav)
                X[i, :, :, 0] = mel
                y[i] = float(lab)
            except Exception:
                # if loading failed, fill zeros and keep label (so training won't crash)
                X[i, :, :, 0] = np.zeros((N_MELS, TIME_STEPS), dtype=np.float32)
                y[i] = float(lab)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.files, self.labels))
            random.shuffle(combined)
            self.files, self.labels = zip(*combined)
            self.files = list(self.files); self.labels = list(self.labels)
