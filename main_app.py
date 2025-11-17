# app_streamlit.py (upgraded UI) — shows "Developed by Alok Kushwaha" prominently
import io
import logging
import os
import traceback
from datetime import datetime
from tempfile import NamedTemporaryFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# -------------------------
# Optional C-extension audio playback (guarded)
# -------------------------
# simpleaudio is a C-extension that often requires system libs (alsa) to build.
# Make it optional in cloud environments where those headers are unavailable.
try:
    import simpleaudio as sa
    _HAS_SIMPLEAUDIO = True
except Exception:
    sa = None
    _HAS_SIMPLEAUDIO = False

# -------------------------
# Who developed this app
# -------------------------
DEV_NAME = "Alok Kushwaha"

# -------------------------
# Tiny logging / debug helpers
# -------------------------
logger = logging.getLogger("deepfake_app")
logger.addHandler(logging.NullHandler())

# -------------------------
# Try to import project helpers (fall back gracefully)
# -------------------------
try:
    from features import extract_features_from_file, safe_read_audio
except Exception as e:
    safe_read_audio = None
    extract_features_from_file = None
    # don't call st.* at import time in some contexts; set a warning later
    features_import_error = str(e)
else:
    features_import_error = None

# -------------------------
# CNN import handler (with error capture) + internal fallback loader
# -------------------------
cnn_import_error = None
load_cnn_fn = None
predict_cnn_file = None
CNN_AVAILABLE = False
cnn_meta = None  # placeholder for metadata (if any)

# First try to import a local cnn_predict.py if present (preferred)
try:
    # prefer project-provided helper if it exists
    from cnn_predict import load_cnn as load_cnn_fn  # type: ignore
    from cnn_predict import predict_file as predict_cnn_file  # type: ignore
    CNN_AVAILABLE = True
except Exception:
    # remember traceback for debug UI
    cnn_import_error = traceback.format_exc()
    load_cnn_fn = None
    predict_cnn_file = None
    CNN_AVAILABLE = False

# If no cnn_predict helper, provide an internal fallback using tensorflow.keras (if available)
if not CNN_AVAILABLE:
    try:
        # attempt to import tensorflow (this may fail on Streamlit cloud if not in requirements)
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

        # internal loader: tries several filenames that exist in your models/ folder
        def _find_cnn_model_file():
            candidates = [
                "models/cnn_audio_fake_detector_final.h5",
                "models/cnn_audio_fake_detector.h5",
                "models/cnn_model.h5",
                "models/cnn.h5",
            ]
            for p in candidates:
                if os.path.exists(p):
                    return p
            # fall back: find any .h5 in models dir
            models_dir = "models"
            if os.path.isdir(models_dir):
                for fname in os.listdir(models_dir):
                    if fname.lower().endswith(".h5"):
                        return os.path.join(models_dir, fname)
            return None

        def load_cnn_internal():
            """
            Loads a Keras model from a guessed filename and optional meta (joblib).
            Returns (model, meta_dict)
            """
            model_path = _find_cnn_model_file()
            meta = {}
            if model_path is None:
                raise FileNotFoundError("No .h5 CNN model file found in models/ (tried common names).")

            # attempt to load metadata if present
            meta_path = "models/cnn_meta.joblib"
            if os.path.exists(meta_path):
                try:
                    meta = joblib.load(meta_path)
                except Exception:
                    meta = {}

            # load model (may raise if TF not present)
            model = keras_load_model(model_path, compile=False)
            return model, meta

        def _make_mel_spectrogram(y, sr, n_mels=128, fmax=8000):
            import librosa
            # Compute mel spectrogram (power)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
            S_db = librosa.power_to_db(S, ref=np.max)
            return S_db  # shape: (n_mels, t)

        def _adapt_spectrogram_to_model_input(S_db, model):
            """
            Given 2D spectrogram S_db (n_mels, t), adapt to model.input_shape:
            - if model expects (None, h, w, c): pad/trim t to w and possibly n_mels -> h.
            - if model expects (None, h, w): similar
            - returns array with batch dim ready for model.predict
            """
            # get target shape from model input
            input_shape = None
            try:
                input_shape = model.input_shape  # e.g. (None, 128, 128, 1)
            except Exception:
                input_shape = None

            # default: try to make shape (1, n_mels, t, 1)
            arr = S_db.astype(np.float32)
            # normalize to [-1,1] or 0-1? keep as dB but scale
            # simple normalization: min-max to [0,1]
            arr = arr - arr.min()
            if arr.max() > 0:
                arr = arr / arr.max()
            # Now adapt dims
            if input_shape is None:
                # fallback: add batch + channel
                return np.expand_dims(np.expand_dims(arr, 0), -1)  # (1, n_mels, t, 1)

            # remove batch dim
            target_shape = list(input_shape)[1:]
            # cases:
            # (h, w, c)  -> conv2d expecting channels-last
            # (h, w)     -> conv2d without channel?
            # if len==3 and last==1 or 3 assume channels last
            if len(target_shape) == 3:
                h_target, w_target, c_target = target_shape
                h_cur, w_cur = arr.shape
                # adjust h (n_mels) -> h_target
                if h_cur < h_target:
                    pad_top = (h_target - h_cur) // 2
                    pad_bottom = h_target - h_cur - pad_top
                    arr = np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode="constant")
                elif h_cur > h_target:
                    # crop center
                    start = (h_cur - h_target) // 2
                    arr = arr[start : start + h_target, :]

                # adjust w (time) -> w_target
                h_cur, w_cur = arr.shape
                if w_cur < w_target:
                    pad_left = (w_target - w_cur) // 2
                    pad_right = w_target - w_cur - pad_left
                    arr = np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="constant")
                elif w_cur > w_target:
                    start = (w_cur - w_target) // 2
                    arr = arr[:, start : start + w_target]

                # add channel dim
                if c_target == 1:
                    arr = np.expand_dims(arr, -1)  # (h_target, w_target, 1)
                else:
                    # if model expects 3 channels, repeat the single channel
                    arr = np.stack([arr] * c_target, axis=-1)

                return np.expand_dims(arr.astype(np.float32), 0)  # add batch dim

            elif len(target_shape) == 2:
                # target (h, w) - no channel dimension
                h_target, w_target = target_shape
                h_cur, w_cur = arr.shape
                # adapt like above
                if h_cur < h_target:
                    pad_top = (h_target - h_cur) // 2
                    pad_bottom = h_target - h_cur - pad_top
                    arr = np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode="constant")
                elif h_cur > h_target:
                    start = (h_cur - h_target) // 2
                    arr = arr[start : start + h_target, :]
                h_cur, w_cur = arr.shape
                if w_cur < w_target:
                    pad_left = (w_target - w_cur) // 2
                    pad_right = w_target - w_cur - pad_left
                    arr = np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="constant")
                elif w_cur > w_target:
                    start = (w_cur - w_target) // 2
                    arr = arr[:, start : start + w_target]
                return np.expand_dims(arr.astype(np.float32), 0)

            else:
                # unknown input shape: return (1, n_mels, t, 1)
                return np.expand_dims(np.expand_dims(arr, 0), -1)

        def predict_cnn_internal(audio_path, model=None, meta=None):
            """
            Predict using a supplied keras model or by loading it if model is None.
            Returns dict: {"prob_real":float, "prob_fake":float, "raw": <model_output>}
            """
            # load model if needed
            if model is None:
                model, meta = load_cnn_internal()

            # read audio (prefer safe_read_audio if available)
            if safe_read_audio is not None:
                y_sr = safe_read_audio(audio_path, target_sr=None, mono=True)
                if y_sr is None:
                    raise RuntimeError("safe_read_audio failed to read audio")
                y, sr = y_sr
            else:
                import librosa
                y, sr = librosa.load(audio_path, sr=None, mono=True)

            # short trim/pad: ensure non-empty
            if len(y) == 0:
                raise ValueError("Empty audio")

            # compute mel
            n_mels = int(meta.get("n_mels", 128))
            fmax = int(meta.get("fmax", 8000))
            S_db = _make_mel_spectrogram(y, sr, n_mels=n_mels, fmax=fmax)

            # adapt to model input
            X = _adapt_spectrogram_to_model_input(S_db, model)

            # model predict (ensure float32)
            preds = model.predict(X)
            # Try to interpret predictions:
            # If model outputs single sigmoid -> [ [p_fake] ] or [ [p_real] ]
            pred_val = None
            if isinstance(preds, (list, tuple)):
                # some models return multiple outputs; choose first
                preds = preds[0]

            # preds shape: (1, n) or (1,)
            try:
                preds_arr = np.array(preds).reshape(preds.shape[0], -1)
            except Exception:
                preds_arr = np.array(preds)
                if preds_arr.ndim == 0:
                    preds_arr = preds_arr.reshape(1, 1)

            # default interpretation:
            # If binary classification with 1 output (sigmoid): preds_arr[0,0] = prob_fake (or prob_real) depending on training.
            # If softmax with 2 outputs: preds_arr[0,1] is prob_real if class order [fake, real] or vice versa.
            prob_real = None
            prob_fake = None

            if preds_arr.shape[1] == 1:
                # single probability — we don't know whether it's prob_fake or prob_real.
                p = float(preds_arr[0, 0])
                # Heuristic: many models output prob of positive class, where positive=1 often mapped to "real".
                # Try to use meta if available
                map_positive = meta.get("positive_class", "real") if isinstance(meta, dict) else "real"
                if map_positive == "fake":
                    prob_fake = p
                    prob_real = 1.0 - p
                else:
                    prob_real = p
                    prob_fake = 1.0 - p
            elif preds_arr.shape[1] >= 2:
                # assume softmax [prob_fake, prob_real] or [prob_real, prob_fake]
                # try to use meta.class_order if present
                class_order = meta.get("class_order") if isinstance(meta, dict) else None
                if class_order and isinstance(class_order, (list, tuple)) and len(class_order) >= 2:
                    try:
                        idx_real = class_order.index("real")
                        idx_fake = class_order.index("fake")
                        prob_real = float(preds_arr[0, idx_real])
                        prob_fake = float(preds_arr[0, idx_fake])
                    except Exception:
                        # fallback: take index 1 as real
                        prob_real = float(preds_arr[0, 1])
                        prob_fake = float(preds_arr[0, 0])
                else:
                    # fallback: assume index 1 is real
                    prob_real = float(preds_arr[0, 1])
                    prob_fake = float(preds_arr[0, 0])

            else:
                # last-resort
                prob_real = float(preds_arr.flatten()[0])
                prob_fake = 1.0 - prob_real

            return {"prob_real": float(prob_real), "prob_fake": float(prob_fake), "raw": preds}

        # assign fallback functions
        load_cnn_fn = load_cnn_internal
        predict_cnn_file = predict_cnn_internal
        CNN_AVAILABLE = True
        cnn_import_error = None

    except Exception as e:
        # tensorflow not available or other error — CNN will be unavailable; capture traceback
        cnn_import_error = traceback.format_exc()
        load_cnn_fn = None
        predict_cnn_file = None
        CNN_AVAILABLE = False

# If we have a loader, attempt to pre-load meta if possible (will be used later)
if load_cnn_fn is not None:
    try:
        # load model (but don't keep heavy model in memory if not needed)
        # We will just get meta if the loader returns meta
        mload = None
        try:
            mload = load_cnn_fn()
        except Exception:
            mload = None
        if isinstance(mload, tuple) and len(mload) >= 2:
            # (model, meta)
            _, cnn_meta = mload
        else:
            cnn_meta = {}
    except Exception:
        cnn_meta = {}

# -------------------------
# Page config & tiny CSS
# -------------------------
# include developer name in browser tab title and a small icon
st.set_page_config(page_title=f"{DEV_NAME} — Audio Deepfake Detector", layout="wide", initial_sidebar_state="expanded", page_icon="🎧")

# light styling
st.markdown(
    """
    <style>
      .stApp { font-family: "Segoe UI", Roboto, Arial; }
      .header-title { font-size:30px; font-weight:700; }
      .muted { color: #9aa3ad; }
      .card { background: #0f1720; padding: 12px; border-radius: 10px; }
      .small { font-size:12px; color:#9aa3ad; }
      .dev-badge { font-size:12px; color:#8a8a8a; text-align:right; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Sidebar: settings, debug toggle & model snapshot
# ----------------------
st.sidebar.header("Settings & Models")
model_choice = st.sidebar.radio("Model", options=["Auto", "RandomForest", "CNN", "Ensemble", "Both"], index=0)
threshold = st.sidebar.slider("Fake threshold (prob_fake ≥ threshold)", 0.0, 1.0, 0.50, 0.01)
st.sidebar.markdown("---")

# Debug toggle (small enhancement)
DEBUG = st.sidebar.checkbox("Enable debug logs", value=False)
if DEBUG:
    # show a small logging area
    st.sidebar.markdown("**Debug logs:**")
    debug_box = st.sidebar.empty()
else:
    debug_box = None

# ----------------------
# Helpers & model loads
# ----------------------
@st.cache_resource
def load_rf_model(path="models/rf_model.joblib"):
    if not os.path.exists(path):
        return None, None
    data = joblib.load(path)
    model = data.get("model") if isinstance(data, dict) and "model" in data else data
    scaler = data.get("scaler") if isinstance(data, dict) and "scaler" in data else None
    return model, scaler

@st.cache_resource
def load_cnn_model_cached():
    """
    Use the previously discovered load_cnn_fn (either project-provided or internal fallback).
    Return (model, meta) or (None, None)
    """
    if load_cnn_fn is None:
        return None, None
    try:
        loaded = load_cnn_fn()
        if isinstance(loaded, tuple):
            return loaded[0], (loaded[1] if len(loaded) > 1 else {})
        return loaded, {}
    except Exception:
        # preserve trace in debug
        if DEBUG and debug_box:
            debug_box.text(traceback.format_exc())
        return None, None

@st.cache_resource
def load_ensemble_cached(path="models/ensemble_meta.joblib"):
    if not os.path.exists(path):
        return None
    try:
        return load_ensemble_fn(path) if load_ensemble_fn is not None else None
    except Exception:
        return None

def rf_predict_proba(model, scaler, feats):
    """Return (prob_real, prob_fake) or (None, None) if failed"""
    if model is None or feats is None:
        return None, None
    X = feats.reshape(1, -1)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes and 0 in classes:
            idx_real = classes.index(1)
            idx_fake = classes.index(0)
            prob_real = float(proba_arr[0, idx_real])
            prob_fake = float(proba_arr[0, idx_fake])
        else:
            try:
                prob_real = float(proba_arr[0, 1])
                prob_fake = 1.0 - prob_real
            except Exception:
                prob_fake = float(proba_arr[0, 0])
                prob_real = 1.0 - prob_fake
        return prob_real, prob_fake
    else:
        pred = int(model.predict(X)[0])
        prob_real = 1.0 if pred == 1 else 0.0
        prob_fake = 1.0 - prob_real
        return prob_real, prob_fake

def plot_wave_mel(y, sr, title_prefix="", ax_wf=None, ax_mel=None):
    import librosa
    import librosa.display

    # create fresh axes if None
    created_fig = False
    if ax_wf is None or ax_mel is None:
        fig, (ax_wf, ax_mel) = plt.subplots(2, 1, figsize=(9, 4))
        created_fig = True
    ax_wf.clear(); ax_mel.clear()
    times = np.arange(len(y)) / sr if sr and len(y) else np.array([0])
    ax_wf.plot(times, y, linewidth=0.6)
    ax_wf.set_title(f"{title_prefix} — Waveform", fontsize=10)
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        im = ax_mel.imshow(S_db, origin="lower", aspect="auto")
        ax_mel.set_title(f"{title_prefix} — Mel-spectrogram (dB)", fontsize=10)
    except Exception as e:
        ax_mel.text(0.5, 0.5, "Mel spectrogram failed", ha="center")
    if created_fig:
        plt.tight_layout()
        return fig
    return None

# ----------------------
# Layout: header + sidebar
# ----------------------
# top header
col_h1, col_h2 = st.columns([0.9, 0.1])
with col_h1:
    # optional logo if exists
    logo_path = "images/ai.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=54)
    st.markdown('<div class="header-title">Audio Deepfake Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload audio, run models (RF / CNN / Ensemble), compare results and export history.</div>', unsafe_allow_html=True)

with col_h2:
    st.markdown(f"<div class='dev-badge'>Developed by {DEV_NAME}</div>", unsafe_allow_html=True)

# Model availability snapshot
with st.spinner("Loading models and checking availability..."):
    rf_model, rf_scaler = load_rf_model()
    cnn_model, cnn_meta = load_cnn_model_cached()
    ensemble_obj = load_ensemble_cached()

st.sidebar.markdown("**Models available**")

# Show CNN import errors visibly
if cnn_import_error:
    st.sidebar.error("CNN module could not be loaded.")
    if DEBUG and debug_box:
        debug_box.text(cnn_import_error)
    else:
        st.sidebar.caption("Enable debug logs to see the full exception. (Likely missing TensorFlow or wrong model path.)")

st.sidebar.write(f"- RandomForest: {'✅' if rf_model is not None else '❌'}")
st.sidebar.write(f"- CNN: {'✅' if cnn_model is not None else '❌'}")
st.sidebar.write(f"- Ensemble: {'✅' if ensemble_obj is not None else '❌'}")
st.sidebar.markdown("---")

# Add developer mention in sidebar
st.sidebar.markdown(f"**Built & maintained by:** {DEV_NAME}")

if features_import_error:
    st.sidebar.error("features.py import failed. Some functionality will be disabled.")
    st.sidebar.caption(features_import_error)
    if DEBUG and debug_box:
        debug_box.text(features_import_error)

# ----------------------
# Main UI: uploader + action + visuals
# ----------------------
left_col, right_col = st.columns([1.4, 0.9])

with left_col:
    uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg", "m4a"])
    st.markdown("**Quick tips:** trim long files to <30s for faster results.")
    run_btn = st.button("Run Prediction", key="run_btn")
    plot_placeholder = st.empty()
    audio_player_placeholder = st.empty()

with right_col:
    status_box = st.empty()
    st.markdown("### Results")
    rf_card = st.empty()
    cnn_card = st.empty()
    ens_card = st.empty()
    comp_card = st.empty()

# session history initialization
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------
# Utility helpers
# ----------------------
def save_temp_file(uploaded_file):
    tf = NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tf.write(uploaded_file.read())
    tf.flush()
    tf.close()
    return tf.name

def make_history_row(model_name, path, prob_real, prob_fake, label):
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": os.path.basename(path),
        "model": model_name,
        "label": label,
        "prob_real": float(prob_real) if prob_real is not None else None,
        "prob_fake": float(prob_fake) if prob_fake is not None else None,
    }

def show_probability_bar(container, prob_real, prob_fake, label_text="Result"):
    """Render a small card with metrics and progress bars."""
    prob_real = 0.0 if prob_real is None else prob_real
    prob_fake = 0.0 if prob_fake is None else prob_fake
    color = "#7BE495" if prob_fake < threshold else "#FF6B6B"
    with container.container():
        st.markdown(f"**{label_text}**")
        st.metric(label="Label", value=("Fake" if prob_fake >= threshold else "Real"))
        st.write(f"prob_real: **{prob_real:.4f}** — prob_fake: **{prob_fake:.4f}**")
        try:
            st.progress(int(prob_real * 100))
            st.progress(int(prob_fake * 100))
        except Exception:
            pass
        st.markdown("")

# ----------------------
# Main prediction flow
# ----------------------
if run_btn:
    if uploaded is None:
        st.warning("Please upload an audio file first.")
    else:
        audio_path = save_temp_file(uploaded)
        try:
            with st.spinner("Loading audio and models..."):
                # load audio (prefer safe_read_audio)
                if safe_read_audio is not None:
                    y_sr = safe_read_audio(audio_path, target_sr=None, mono=True)
                    if y_sr is None:
                        st.error("Failed to load audio with safe_read_audio().")
                        raise RuntimeError("safe_read_audio failed")
                    y, sr = y_sr
                else:
                    import librosa
                    y, sr = librosa.load(audio_path, sr=None, mono=True)

                audio_player_placeholder.audio(audio_path)
                fig = plot_wave_mel(y, sr, title_prefix="File")
                plot_placeholder.pyplot(fig)

                # decide which models to run
                chosen = model_choice
                if chosen == "Auto":
                    if rf_model is not None:
                        chosen = "RandomForest"
                    elif cnn_model is not None:
                        chosen = "CNN"
                    elif ensemble_obj is not None:
                        chosen = "Ensemble"

                rf_card.empty(); cnn_card.empty(); ens_card.empty(); comp_card.empty()
                status_box.info("Running selected model(s)...")

            # run RF if requested / available
            rf_res = None
            if chosen in ("RandomForest", "Both"):
                if rf_model is None:
                    rf_card.info("RandomForest model not available.")
                else:
                    with st.spinner("Running RandomForest..."):
                        feats = None
                        if extract_features_from_file is not None:
                            feats = extract_features_from_file(audio_path)
                        if feats is None:
                            rf_card.error("RF: feature extraction failed.")
                        else:
                            prob_real_rf, prob_fake_rf = rf_predict_proba(rf_model, rf_scaler, feats)
                            label_rf = "Fake" if prob_fake_rf >= threshold else "Real"
                            rf_res = {"prob_real": prob_real_rf, "prob_fake": prob_fake_rf, "label": label_rf}
                            show_probability_bar(rf_card, prob_real_rf, prob_fake_rf, label_text="RandomForest")

            # run CNN if requested / available
            cnn_res = None
            if chosen in ("CNN", "Both"):
                if cnn_model is None or predict_cnn_file is None:
                    cnn_card.info("CNN model or helper not available.")
                    if cnn_import_error and DEBUG and debug_box:
                        debug_box.text(cnn_import_error)
                else:
                    with st.spinner("Running CNN..."):
                        try:
                            # if cnn_model already loaded from cache, pass it; otherwise predict_cnn_file will load it
                            res = predict_cnn_file(audio_path, model=cnn_model, meta=cnn_meta) if cnn_model is not None else predict_cnn_file(audio_path)
                            prob_real_cnn = float(res.get("prob_real", 0.0))
                            prob_fake_cnn = float(res.get("prob_fake", 1.0 - prob_real_cnn))
                            label_cnn = "Fake" if prob_fake_cnn >= threshold else "Real"
                            cnn_res = {"prob_real": prob_real_cnn, "prob_fake": prob_fake_cnn, "label": label_cnn}
                            show_probability_bar(cnn_card, prob_real_cnn, prob_fake_cnn, label_text="CNN")
                        except Exception as e:
                            cnn_card.error(f"CNN failed: {e}")
                            if DEBUG and debug_box:
                                debug_box.text(traceback.format_exc())

            # run Ensemble if requested
            ens_res = None
            if chosen == "Ensemble":
                if ensemble_obj is None or predict_ensemble_file is None:
                    ens_card.info("Ensemble model/helper not available.")
                else:
                    with st.spinner("Running Ensemble..."):
                        try:
                            res = predict_ensemble_file(audio_path, ensemble_obj)
                            prob_real_e = float(res.get("prob_real", 0.0))
                            prob_fake_e = float(res.get("prob_fake", 1.0 - prob_real_e))
                            label_e = "Fake" if prob_fake_e >= threshold else "Real"
                            ens_res = {"prob_real": prob_real_e, "prob_fake": prob_fake_e, "label": label_e}
                            show_probability_bar(ens_card, prob_real_e, prob_fake_e, label_text="Ensemble")
                        except Exception as e:
                            ens_card.error(f"Ensemble failed: {e}")
                            if DEBUG and debug_box:
                                debug_box.text(traceback.format_exc())

            # combine results for "Both"
            if chosen == "Both":
                with st.spinner("Computing comparison..."):
                    if rf_res is not None:
                        rf_text = f"RF → {rf_res['label']} (real:{rf_res['prob_real']:.3f}, fake:{rf_res['prob_fake']:.3f})"
                    else:
                        rf_text = "RF → N/A"
                    if cnn_res is not None:
                        cnn_text = f"CNN → {cnn_res['label']} (real:{cnn_res['prob_real']:.3f}, fake:{cnn_res['prob_fake']:.3f})"
                    else:
                        cnn_text = "CNN → N/A"

                    if rf_res and cnn_res:
                        avg_real = (rf_res['prob_real'] + cnn_res['prob_real']) / 2.0
                        avg_fake = 1.0 - avg_real
                        agree = (rf_res['label'] == cnn_res['label'])
                        agree_text = "✅ Agreement" if agree else "⚠️ Disagreement"
                        final_label = rf_res['label'] if agree else f"Conflict ({rf_res['label']}/{cnn_res['label']})"
                        comp_card.success(f"**Final:** {final_label}\n\n{agree_text}\n\nRF real:{rf_res['prob_real']:.3f}, CNN real:{cnn_res['prob_real']:.3f}\nAvg real:{avg_real:.3f}  Avg fake:{avg_fake:.3f}")
                        st.session_state.history.append(make_history_row("Both", audio_path, avg_real, avg_fake, final_label))
                    else:
                        final_label = rf_res['label'] if rf_res else (cnn_res['label'] if cnn_res else "N/A")
                        prob_r = rf_res['prob_real'] if rf_res else (cnn_res['prob_real'] if cnn_res else None)
                        prob_f = rf_res['prob_fake'] if rf_res else (cnn_res['prob_fake'] if cnn_res else None)
                        comp_card.info(f"Final: {final_label}")
                        st.session_state.history.append(make_history_row("Both", audio_path, prob_r, prob_f, final_label))
            else:
                if rf_res is not None and chosen == "RandomForest":
                    st.session_state.history.append(make_history_row("RandomForest", audio_path, rf_res['prob_real'], rf_res['prob_fake'], rf_res['label']))
                if cnn_res is not None and chosen == "CNN":
                    st.session_state.history.append(make_history_row("CNN", audio_path, cnn_res['prob_real'], cnn_res['prob_fake'], cnn_res['label']))
                if ens_res is not None and chosen == "Ensemble":
                    st.session_state.history.append(make_history_row("Ensemble", audio_path, ens_res['prob_real'], ens_res['prob_fake'], ens_res['label']))

            status_box.success("Prediction finished.")
        except Exception as e:
            status_box.error(f"Prediction failed: {e}")
            if DEBUG and debug_box:
                debug_box.text(traceback.format_exc())

# ----------------------
# History & export
# ----------------------
st.markdown("---")
with st.expander("📜 Prediction history (session)"):
    hist = st.session_state.get("history", [])
    if hist:
        import pandas as pd
        df = pd.DataFrame(hist)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="prediction_history.csv", mime="text/csv")
        st.write("")  # spacing
        if st.button("Clear history"):
            st.session_state.history = []
            st.experimental_rerun()
    else:
        st.write("No predictions yet — run the model to populate history.")

# small footer — clearly credit you as developer
st.markdown(
    f"<div style='text-align:center;color:#8a8a8a;font-size:12px;'>© {datetime.now().year} — Developed by {DEV_NAME}</div>",
    unsafe_allow_html=True
)
