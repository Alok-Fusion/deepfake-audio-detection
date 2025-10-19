# app_streamlit.py (upgraded UI) — shows "Developed by Alok Kushwaha" prominently
import io
import os
from datetime import datetime
from tempfile import NamedTemporaryFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# -------------------------
# Who developed this app
# -------------------------
DEV_NAME = "Alok Kushwaha"

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

# optional: cnn/ensemble helpers if present in your repo
try:
    from cnn_predict import load_cnn as load_cnn_fn
    from cnn_predict import predict_file as predict_cnn_file
    CNN_AVAILABLE = True
except Exception:
    load_cnn_fn = None
    predict_cnn_file = None
    CNN_AVAILABLE = False

try:
    from ensemble_predict import load_ensemble as load_ensemble_fn
    from ensemble_predict import predict_file as predict_ensemble_file
    ENSEMBLE_AVAILABLE = True
except Exception:
    load_ensemble_fn = None
    predict_ensemble_file = None
    ENSEMBLE_AVAILABLE = False

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
    if not CNN_AVAILABLE:
        return None, None
    try:
        return load_cnn_fn()
    except Exception:
        return None, None

@st.cache_resource
def load_ensemble_cached(path="models/ensemble_meta.joblib"):
    if not ENSEMBLE_AVAILABLE and not os.path.exists(path):
        return None
    try:
        return load_ensemble_fn(path)
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
    st.markdown('<div class="header-title">🎙️ Audio Deepfake Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload audio, run models (RF / CNN / Ensemble), compare results and export history.</div>', unsafe_allow_html=True)

with col_h2:
    # small right aligned developer badge under header area
    st.markdown(f"<div class='dev-badge'>Developed by {DEV_NAME}</div>", unsafe_allow_html=True)

# settings in sidebar
st.sidebar.header("Settings & Models")
model_choice = st.sidebar.radio("Model", options=["Auto", "RandomForest", "CNN", "Ensemble", "Both"], index=0)
threshold = st.sidebar.slider("Fake threshold (prob_fake ≥ threshold)", 0.0, 1.0, 0.50, 0.01)
st.sidebar.markdown("---")

# Model availability snapshot
rf_model, rf_scaler = load_rf_model()
cnn_model, cnn_meta = load_cnn_model_cached()
ensemble_obj = load_ensemble_cached()
st.sidebar.markdown("**Models available**")
st.sidebar.write(f"- RandomForest: {'✅' if rf_model is not None else '❌'}")
st.sidebar.write(f"- CNN: {'✅' if cnn_model is not None else '❌'}")
st.sidebar.write(f"- Ensemble: {'✅' if ensemble_obj is not None else '❌'}")
st.sidebar.markdown("---")

# Add developer mention in sidebar
st.sidebar.markdown(f"**Built & maintained by:** {DEV_NAME}")

if features_import_error:
    st.sidebar.error("features.py import failed. Some functionality will be disabled.")
    st.sidebar.caption(features_import_error)

# ----------------------
# Main UI: uploader + action + visuals
# ----------------------
left_col, right_col = st.columns([1.4, 0.9])

with left_col:
    uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg", "m4a"])
    # quick actions
    st.markdown("**Quick tips:** trim long files to <30s for faster results.")
    run_btn = st.button("▶️ Run Prediction", key="run_btn")

    # visual placeholder
    plot_placeholder = st.empty()
    audio_player_placeholder = st.empty()

with right_col:
    # status & model cards
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
    # decide color based on label_text or prob_fake
    color = "#7BE495" if prob_fake < threshold else "#FF6B6B"
    # Render
    with container.container():
        st.markdown(f"**{label_text}**")
        st.metric(label="Label", value=("Fake" if prob_fake >= threshold else "Real"))
        st.write(f"prob_real: **{prob_real:.4f}** — prob_fake: **{prob_fake:.4f}**")
        # progress bars (two small)
        try:
            st.progress(int(prob_real * 100))
            st.progress(int(prob_fake * 100))
        except Exception:
            # some streamlit versions don't support two progress bars sequentially; ignore if fails
            pass
        # small spacer
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
                # load audio
                if safe_read_audio is not None:
                    y_sr = safe_read_audio(audio_path, target_sr=None, mono=True)
                    if y_sr is None:
                        st.error("Failed to load audio with safe_read_audio().")
                        raise RuntimeError("safe_read_audio failed")
                    y, sr = y_sr
                else:
                    import librosa
                    y, sr = librosa.load(audio_path, sr=None, mono=True)

                # show audio player & waveform + mel
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

                # reset right side cards
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
                            # show card
                            show_probability_bar(rf_card, prob_real_rf, prob_fake_rf, label_text="RandomForest")
                            st.experimental_rerun() if False else None  # no-op to keep layout consistent

            # run CNN if requested / available
            cnn_res = None
            if chosen in ("CNN", "Both"):
                if cnn_model is None or predict_cnn_file is None:
                    cnn_card.info("CNN model or helper not available.")
                else:
                    with st.spinner("Running CNN..."):
                        try:
                            res = predict_cnn_file(audio_path, model=cnn_model, meta=cnn_meta)
                            prob_real_cnn = float(res.get("prob_real", 0.0))
                            prob_fake_cnn = float(res.get("prob_fake", 1.0 - prob_real_cnn))
                            label_cnn = "Fake" if prob_fake_cnn >= threshold else "Real"
                            cnn_res = {"prob_real": prob_real_cnn, "prob_fake": prob_fake_cnn, "label": label_cnn}
                            show_probability_bar(cnn_card, prob_real_cnn, prob_fake_cnn, label_text="CNN")
                        except Exception as e:
                            cnn_card.error(f"CNN failed: {e}")

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

            # If Both requested, compute comparison + final label
            if chosen == "Both":
                with st.spinner("Computing comparison..."):
                    lines = []
                    if rf_res is not None:
                        lines.append(f"RF → {rf_res['label']} (real:{rf_res['prob_real']:.3f}, fake:{rf_res['prob_fake']:.3f})")
                    else:
                        lines.append("RF → N/A")
                    if cnn_res is not None:
                        lines.append(f"CNN → {cnn_res['label']} (real:{cnn_res['prob_real']:.3f}, fake:{cnn_res['prob_fake']:.3f})")
                    else:
                        lines.append("CNN → N/A")

                    # aggregated decision
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
                # single model history entries
                if rf_res is not None and chosen == "RandomForest":
                    st.session_state.history.append(make_history_row("RandomForest", audio_path, rf_res['prob_real'], rf_res['prob_fake'], rf_res['label']))
                if cnn_res is not None and chosen == "CNN":
                    st.session_state.history.append(make_history_row("CNN", audio_path, cnn_res['prob_real'], cnn_res['prob_fake'], cnn_res['label']))
                if ens_res is not None and chosen == "Ensemble":
                    st.session_state.history.append(make_history_row("Ensemble", audio_path, ens_res['prob_real'], ens_res['prob_fake'], ens_res['label']))

            status_box.success("Prediction finished.")
        except Exception as e:
            status_box.error(f"Prediction failed: {e}")

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
        # clear history button
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
