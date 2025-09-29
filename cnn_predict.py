# cnn_predict.py
"""
Simple helper to run inference with the trained CNN.
Returns a dict: {'prob_real': float, 'prob_fake': float}
"""
import os
import joblib
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from cnn_data import mel_spectrogram, SR, TIME_STEPS, N_MELS

CNN_MODEL_PATH = "models/cnn_audio_fake_detector.h5"          # best checkpoint
CNN_META_PATH = "models/cnn_meta.joblib"

def load_cnn():
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"{CNN_MODEL_PATH} not found. Train the CNN first.")
    model = load_model(CNN_MODEL_PATH, compile=False)
    meta = joblib.load(CNN_META_PATH) if os.path.exists(CNN_META_PATH) else {"sr": SR, "time_steps": TIME_STEPS, "n_mels": N_MELS, "duration": 5.0}
    return model, meta

def predict_file(path, model=None, meta=None):
    if model is None or meta is None:
        model, meta = load_cnn()
    sr = meta.get("sr", SR)
    duration = meta.get("duration", 5.0)
    # load audio, pad/truncate
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    S = mel_spectrogram(y, sr=sr, n_mels=meta.get("n_mels", N_MELS), time_steps=meta.get("time_steps", TIME_STEPS))
    x = S[np.newaxis, :, :, np.newaxis].astype(np.float32)
    prob_real = float(model.predict(x, verbose=0)[0, 0])
    prob_fake = 1.0 - prob_real
    return {"prob_real": prob_real, "prob_fake": prob_fake}
