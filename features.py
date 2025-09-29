# features.py (fixed safe_read_audio to handle target_sr=None)
import os

import librosa
import numpy as np
import soundfile as sf

SR = 22050
MAX_DURATION = 5.0
N_MFCC = 40

def safe_read_audio(path, target_sr=SR, mono=True, duration=MAX_DURATION, res_type='kaiser_fast'):
    """
    Robustly read audio. Accepts target_sr==None (interpreted as module SR).
    Returns (y, sr) or (None, None).
    """
    # If caller passed None, use default SR
    if target_sr is None:
        target_sr = SR

    try:
        # Try soundfile first (fast for PCM WAVs)
        y, sr = sf.read(path, dtype='float32')
        if y is None:
            raise RuntimeError("soundfile returned None")
        if hasattr(y, "ndim") and y.ndim == 2 and mono:
            y = np.mean(y, axis=1)
    except Exception:
        # fallback to librosa which can use audioread/ffmpeg for many formats
        try:
            y, sr = librosa.load(path, sr=target_sr, mono=mono, duration=duration, res_type=res_type)
        except Exception:
            return None, None

    # If soundfile returned a sample rate different than desired, resample
    if sr != target_sr:
        try:
            y = librosa.resample(np.asarray(y, dtype='float32'), orig_sr=sr, target_sr=target_sr, res_type=res_type)
            sr = target_sr
        except Exception:
            # if resample fails, keep original sr
            pass

    # pad/truncate to fixed length for consistent downstream processing
    target_len = int(target_sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return np.asarray(y, dtype='float32'), sr

def extract_mfcc(y, sr, n_mfcc=N_MFCC):
    """Return 1D MFCC vector (mean across time)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def extract_features_from_file(path):
    """Read file and return averaged MFCC vector (1D) or None on failure."""
    y_sr = safe_read_audio(path)
    if y_sr is None:
        return None
    y, sr = y_sr
    try:
        return extract_mfcc(y, sr)
    except Exception:
        return None

def list_files_labels(data_dir):
    """Return (files, labels) expecting data_dir/{fake,real}"""
    files, labels = [], []
    fake_dir = os.path.join(data_dir, "fake")
    real_dir = os.path.join(data_dir, "real")
    if os.path.isdir(fake_dir):
        for f in sorted(os.listdir(fake_dir)):
            if f.lower().endswith(('.wav','.mp3','.flac','.ogg','.m4a')):
                files.append(os.path.join(fake_dir, f)); labels.append(0)
    if os.path.isdir(real_dir):
        for f in sorted(os.listdir(real_dir)):
            if f.lower().endswith(('.wav','.mp3','.flac','.ogg','.m4a')):
                files.append(os.path.join(real_dir, f)); labels.append(1)
    return files, labels
