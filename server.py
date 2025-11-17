# server.py
import matplotlib
matplotlib.use("Agg")  

import base64
import io
import os
import traceback
from typing import Optional

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# Config
MODEL_PATH = os.path.join("models", "rf_model.joblib")
ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

app = FastAPI(title="Audio Deepfake Detector - Backend")

# Allow local dev from file:// or any origin while testing (you can tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html and any static files if present
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# --------- Model loading ----------
_model_cache = {"model": None, "scaler": None, "loaded": False}


def load_model():
    """Load model + scaler (joblib). Cache in memory."""
    if _model_cache["loaded"]:
        return _model_cache["model"], _model_cache["scaler"]
    if not os.path.exists(MODEL_PATH):
        _model_cache["loaded"] = True
        _model_cache["model"] = None
        _model_cache["scaler"] = None
        return None, None
    try:
        data = joblib.load(MODEL_PATH)
        # saved either as dict {'model':..., 'scaler':...} or raw estimator
        if isinstance(data, dict) and "model" in data:
            model = data.get("model")
            scaler = data.get("scaler")
        else:
            model = data
            scaler = None
        _model_cache["model"], _model_cache["scaler"], _model_cache["loaded"] = model, scaler, True
        return model, scaler
    except Exception:
        traceback.print_exc()
        _model_cache["loaded"] = True
        _model_cache["model"] = None
        _model_cache["scaler"] = None
        return None, None


# --------- Feature extraction (use user's features.py if available) ----------
try:
    from features import extract_features_from_file, safe_read_audio  # user-supplied helpers
    _USING_USER_FEATURES = True
except Exception:
    _USING_USER_FEATURES = False

    def safe_read_audio(path: str, target_sr: Optional[int] = None, mono: bool = True):
        """Fallback audio loader using librosa. Returns (y, sr) or None."""
        try:
            y, sr = librosa.load(path, sr=target_sr, mono=mono)
            return y, sr
        except Exception:
            return None

    def extract_features_from_file(path: str):
        """Simple MFCC-based fallback: mean+std of 40 MFCCs."""
        try:
            ysr = safe_read_audio(path, target_sr=22050, mono=True)
            if ysr is None:
                return None
            y, sr = ysr
            # ensure at least a tiny audio
            if y is None or len(y) < 10:
                return None
            mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            # aggregate: mean + std per coefficient
            mf_mean = np.mean(mf, axis=1)
            mf_std = np.std(mf, axis=1)
            feats = np.concatenate([mf_mean, mf_std]).astype(np.float32)
            return feats
        except Exception:
            traceback.print_exc()
            return None


# --------- Helpers ----------
def predict_proba_from_model(model, scaler, feats):
    """Return (prob_real, prob_fake). Handles classes_ order robustly."""
    if model is None or feats is None:
        return None, None
    X = feats.reshape(1, -1)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", []))
        # try to map label 1 -> real, 0 -> fake if present
        try:
            if 1 in classes and 0 in classes:
                idx_real = classes.index(1)
                idx_fake = classes.index(0)
                prob_real = float(proba[0, idx_real])
                prob_fake = float(proba[0, idx_fake])
            else:
                # fallback assume column 1 is real
                prob_real = float(proba[0, 1])
                prob_fake = 1.0 - prob_real
        except Exception:
            # very fallback
            prob_real = float(proba[0, -1])
            prob_fake = 1.0 - prob_real
    else:
        pred = int(model.predict(X)[0])
        prob_real = 1.0 if pred == 1 else 0.0
        prob_fake = 1.0 - prob_real
    return prob_real, prob_fake


def make_spectrogram_base64(y, sr, n_mels=128, fmax=8000):
    """Return data-url PNG of mel spectrogram."""
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(S_db, origin="lower", aspect="auto")
        ax.axis("off")
        buf = io.BytesIO()
        fig.tight_layout(pad=0)
        fig.savefig(buf, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{img_b64}"
    except Exception:
        traceback.print_exc()
        return None


# --------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def root():
    """Serve local index.html if present, otherwise a tiny link page."""
    index_path = "index.html"
    if os.path.exists(index_path):
        return HTMLResponse(open(index_path, "r", encoding="utf-8").read())
    return HTMLResponse(
        "<h3>Audio Deepfake Detector</h3>"
        "<p>Place <code>index.html</code> in the app root to serve the UI.</p>"
        "<p>Or call <code>/predict</code> with a file upload.</p>"
    )


class PredictResponse(BaseModel):
    prob_real: Optional[float]
    prob_fake: Optional[float]
    label: str
    spectrogram: Optional[str] = None  # data-url png


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    """Accept an audio file upload and return prediction + spectrogram (base64 PNG)."""
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")

        # save to temp file
        tmp_path = f"/tmp/{file.filename}"
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        # load model
        model, scaler = load_model()
        if model is None:
            raise HTTPException(status_code=500, detail=f"Model not found at {MODEL_PATH}")

        # extract features
        feats = extract_features_from_file(tmp_path)
        if feats is None:
            raise HTTPException(status_code=400, detail="Feature extraction failed for uploaded file.")

        prob_real, prob_fake = predict_proba_from_model(model, scaler, feats)
        if prob_real is None:
            raise HTTPException(status_code=500, detail="Prediction failed (no probabilities).")

        label = "Fake" if prob_fake >= threshold else "Real"

        # generate spectrogram
        ysr = safe_read_audio(tmp_path, target_sr=None, mono=True)
        spectrogram = None
        if ysr is not None:
            y, sr = ysr
            spectrogram = make_spectrogram_base64(y, sr)

        # cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return PredictResponse(
            prob_real=prob_real,
            prob_fake=prob_fake,
            label=label,
            spectrogram=spectrogram,
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# optional endpoint to check model status
@app.get("/status")
def status():
    model, scaler = load_model()
    return {"model_loaded": model is not None, "uses_user_features": _USING_USER_FEATURES}
