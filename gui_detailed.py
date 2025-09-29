# gui_detailed.py
import csv
import datetime
import os
import threading
import time
import traceback
from tkinter import filedialog

import customtkinter
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

from features import extract_features_from_file, safe_read_audio
from train import DEFAULT_MODEL_PATH as MODEL_PATH
from train import train

# try to import CNN predictor (optional)
try:
    from cnn_predict import load_cnn as load_cnn_fn
    from cnn_predict import predict_file as predict_cnn_file
    CNN_AVAILABLE_IMPORT = True
except Exception:
    load_cnn_fn = None
    predict_cnn_file = None
    CNN_AVAILABLE_IMPORT = False

# optional audio playback (install simpleaudio to enable)
try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except Exception:
    SIMPLEAUDIO_AVAILABLE = False

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

APP_TITLE = "Deepfake Audio Detection (Detailed UI)"
app = customtkinter.CTk()
app.geometry("1200x760")
app.title(APP_TITLE)

# ---------- helpers ----------
HISTORY = []  # list of dicts: timestamp, file, prob_real, prob_fake, label, model

def load_model():
    """Load RandomForest model + scaler from MODEL_PATH (joblib) and return (model, scaler)."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        data = joblib.load(MODEL_PATH)
        model = data.get('model') if isinstance(data, dict) and 'model' in data else data
        scaler = data.get('scaler') if isinstance(data, dict) and 'scaler' in data else None
        return model, scaler
    except Exception as e:
        print("Failed to load RF model:", e)
        return None, None

# CNN cache & loader
cnn_cache = {"model": None, "meta": None, "loaded": False}

def load_cnn_model():
    """
    Try to load CNN model via cnn_predict.load_cnn (if import succeeded).
    Caches result in cnn_cache.
    Returns (model, meta) or (None, None) if not available.
    """
    if not CNN_AVAILABLE_IMPORT:
        return None, None
    if cnn_cache.get("loaded"):
        return cnn_cache["model"], cnn_cache["meta"]
    try:
        model, meta = load_cnn_fn()
        cnn_cache["model"], cnn_cache["meta"], cnn_cache["loaded"] = model, meta, True
        return model, meta
    except Exception as e:
        print("Failed to load CNN model:", e)
        cnn_cache["model"], cnn_cache["meta"], cnn_cache["loaded"] = None, None, True
        return None, None

def model_info_text(model):
    """Return a short text description for the RF model (if provided)."""
    if model is None:
        return "No RF model loaded."
    lines = []
    lines.append(f"Model class: {model.__class__.__name__}")
    try:
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None:
            lines.append(f"n_features_in_: {n_features}")
    except Exception:
        pass
    # RF specifics
    if model.__class__.__name__.lower().startswith("randomforest"):
        try:
            lines.append(f"n_estimators: {len(model.estimators_)}")
        except Exception:
            pass
    # show classes_ if present
    try:
        classes = getattr(model, "classes_", None)
        if classes is not None:
            lines.append(f"classes_: {list(classes)}")
    except Exception:
        pass
    return "\n".join(lines)

def cnn_info_text(meta):
    if meta is None:
        return "No CNN model loaded."
    lines = []
    lines.append("CNN metadata:")
    for k,v in meta.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)

def pretty_prob(p):
    return f"{p*100:.2f}%"

# ---------- Layout ----------
# top: title
frame_top = customtkinter.CTkFrame(master=app)
frame_top.pack(padx=12, pady=8, fill="x")

logo_path = os.path.join("images", "ai.png")
logo_img = None
if os.path.exists(logo_path):
    logo_img = customtkinter.CTkImage(Image.open(logo_path), size=(48,48))

title = customtkinter.CTkLabel(master=frame_top, text="  Deepfake Audio Detection", image=logo_img,
                              compound="left", font=customtkinter.CTkFont(size=20, weight="bold"))
title.pack(anchor="w")

# main left/right frames
frame_main = customtkinter.CTkFrame(master=app)
frame_main.pack(fill="both", expand=True, padx=12, pady=8)

frame_left = customtkinter.CTkFrame(master=frame_main)
frame_left.pack(side="left", fill="both", expand=True, padx=(0,8))

frame_right = customtkinter.CTkFrame(master=frame_main, width=360)
frame_right.pack(side="right", fill="y")

# --- Left: visualizations and file controls ---
frame_file = customtkinter.CTkFrame(master=frame_left)
frame_file.pack(fill="x", pady=(0,8))

lbl_file = customtkinter.CTkLabel(master=frame_file, text="Select audio file:")
lbl_file.grid(row=0, column=0, padx=8, pady=6, sticky="w")

entry_file = customtkinter.CTkEntry(master=frame_file, width=520)
entry_file.grid(row=0, column=1, padx=6, pady=6)

def browse_file():
    p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a")])
    if p:
        entry_file.delete(0, "end"); entry_file.insert(0, p)
        clear_result_display()

btn_browse = customtkinter.CTkButton(master=frame_file, text="Browse", command=browse_file, width=80)
btn_browse.grid(row=0, column=2, padx=6)

# Run / Train buttons + model selector
frame_ops = customtkinter.CTkFrame(master=frame_left)
frame_ops.pack(fill="x", pady=(0,8))

btn_run = customtkinter.CTkButton(master=frame_ops, text="Predict (Single)", width=140)
btn_run.grid(row=0, column=0, padx=8, pady=6)

btn_run_both = customtkinter.CTkButton(master=frame_ops, text="Run Both (RF + CNN)", width=160)
btn_run_both.grid(row=0, column=1, padx=8, pady=6)

btn_train = customtkinter.CTkButton(master=frame_ops, text="Train RF (from data/)", width=140)
btn_train.grid(row=0, column=2, padx=8, pady=6)

# model selector: Auto / RandomForest / CNN
model_choice_var = customtkinter.StringVar(value="Auto")
model_choice = customtkinter.CTkOptionMenu(master=frame_ops, values=["Auto", "RandomForest", "CNN"], variable=model_choice_var)
model_choice.grid(row=0, column=4, padx=8, pady=6)
model_choice.set("Auto")

# threshold slider
threshold_var = customtkinter.DoubleVar(value=0.50)
threshold_label = customtkinter.CTkLabel(master=frame_ops, text=f"Decision threshold: {threshold_var.get():.2f}")
threshold_label.grid(row=0, column=3, padx=8)
def on_threshold_change(val):
    threshold_label.configure(text=f"Decision threshold: {float(val):.2f}")
threshold_slider = customtkinter.CTkSlider(master=frame_ops, from_=0.0, to=1.0, number_of_steps=100,
                                           command=lambda v: (threshold_var.set(v), on_threshold_change(v)))
threshold_slider.set(0.50)
threshold_slider.grid(row=1, column=3, padx=8, pady=(0,6), sticky="we")

# visual area (matplotlib canvas) — 2 columns x 2 rows: RF waveform+mel (left), CNN waveform+mel (right)
fig, axes = plt.subplots(2, 2, figsize=(10,6), dpi=100)
(ax_rf_wf, ax_cnn_wf), (ax_rf_mel, ax_cnn_mel) = axes
plt.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=frame_left)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True, pady=(4,0))

# result display area (label + prob + comparison)
frame_result = customtkinter.CTkFrame(master=frame_left)
frame_result.pack(fill="x", pady=8)

result_label_var = customtkinter.StringVar(value="No prediction yet")
result_label = customtkinter.CTkLabel(master=frame_result, textvariable=result_label_var, font=customtkinter.CTkFont(size=16, weight="bold"))
result_label.grid(row=0, column=0, padx=8, pady=4, sticky="w")

prob_label_var = customtkinter.StringVar(value="")
prob_label = customtkinter.CTkLabel(master=frame_result, textvariable=prob_label_var)
prob_label.grid(row=1, column=0, padx=8, pady=(0,6), sticky="w")

# comparison area for Run Both
comparison_var = customtkinter.StringVar(value="")
comparison_label = customtkinter.CTkLabel(master=frame_result, textvariable=comparison_var, justify="left")
comparison_label.grid(row=0, column=1, padx=8, pady=4, sticky="e")

# Play button (optional)
def play_audio_bytes(audio_path):
    if not SIMPLEAUDIO_AVAILABLE:
        return "simpleaudio not installed"
    try:
        y,sr = safe_read_audio(audio_path, target_sr=None, mono=True)
        if y is None:
            return "Cannot play: load failed"
        # write to WAV bytes (16-bit) for simpleaudio
        import io

        import soundfile as sf
        bio = io.BytesIO()
        sf.write(bio, y, sr, format='WAV', subtype='PCM_16')
        audio_data = bio.getvalue()
        play_obj = sa.play_buffer(audio_data, 1, 2, sr)
        return "Playing"
    except Exception as e:
        print("Play error:", e)
        return "Play failed"

btn_play = customtkinter.CTkButton(master=frame_result, text="Play (optional)", width=120,
                                  command=lambda: print(play_audio_bytes(entry_file.get().strip())))
btn_play.grid(row=0, column=2, padx=6, pady=4)

# --- Right: model info, history, save ---
frame_right_top = customtkinter.CTkFrame(master=frame_right)
frame_right_top.pack(fill="x", pady=(0,8), padx=8)

lbl_model = customtkinter.CTkLabel(master=frame_right_top, text="Model info", font=customtkinter.CTkFont(size=14, weight="bold"))
lbl_model.pack(anchor="w", pady=(2,6))

model_info_textvar = customtkinter.StringVar(value="No model loaded.")
txt_model_info = customtkinter.CTkLabel(master=frame_right_top, textvariable=model_info_textvar, justify="left")
txt_model_info.pack(anchor="w")

btn_reload_model = customtkinter.CTkButton(master=frame_right_top, text="Reload model(s)", command=lambda: refresh_model_info())
btn_reload_model.pack(pady=(6,0))

# history scrollable frame (colored rows)
frame_hist = customtkinter.CTkFrame(master=frame_right)
frame_hist.pack(fill="both", expand=True, padx=8, pady=(8,8))

lbl_hist = customtkinter.CTkLabel(master=frame_hist, text="Prediction history", font=customtkinter.CTkFont(size=14, weight="bold"))
lbl_hist.pack(anchor="w", pady=(2,6))

hist_scroll = customtkinter.CTkScrollableFrame(master=frame_hist, width=320, height=360)
hist_scroll.pack(fill="both", expand=True, padx=4)

frame_hist_buttons = customtkinter.CTkFrame(master=frame_hist)
frame_hist_buttons.pack(fill="x", pady=(6,0))
btn_save_hist = customtkinter.CTkButton(master=frame_hist_buttons, text="Save history CSV", command=lambda: save_history_csv())
btn_save_hist.pack(side="left", padx=6)
btn_clear_hist = customtkinter.CTkButton(master=frame_hist_buttons, text="Clear history", command=lambda: clear_history())
btn_clear_hist.pack(side="left", padx=6)

status_label = customtkinter.CTkLabel(master=app, text="")
status_label.pack(pady=(4,6))

# ---------- functions ----------
model_cache = {"model": None, "scaler": None}
def refresh_model_info():
    """Reload both RF and CNN info and display summary in the model info panel."""
    model, scaler = load_model()
    model_cache["model"], model_cache["scaler"] = model, scaler

    # try load cnn meta (only meta to show info)
    cnn_model, cnn_meta = load_cnn_model()
    info_lines = []
    if model is not None:
        info_lines.append("RandomForest:")
        info_lines.append(model_info_text(model))
    else:
        info_lines.append("RandomForest: (not found)")

    info_lines.append("")  # spacer

    if cnn_model is not None or cnn_cache.get("loaded"):
        # if loaded successfully, show meta, else indicate not present
        if cnn_meta is not None:
            info_lines.append("CNN:")
            info_lines.append(cnn_info_text(cnn_meta))
        else:
            info_lines.append("CNN: (no meta or not loaded)")
    else:
        info_lines.append("CNN: (not found)")

    model_info_textvar.set("\n".join(info_lines))

def clear_result_display():
    result_label_var.set("No prediction yet")
    prob_label_var.set("")
    comparison_var.set("")
    ax_rf_wf.clear(); ax_rf_mel.clear(); ax_cnn_wf.clear(); ax_cnn_mel.clear(); canvas.draw()

def _add_history_row_widget(entry):
    """Add a colored row to hist_scroll representing entry."""
    # choose color by model
    m = entry.get("model", "RandomForest")
    if m == "RandomForest":
        bg = "#1f6feb20"  # semi-transparent blue tint (CTk will accept hex)
    elif m == "CNN":
        bg = "#1fbf4a20"  # green tint
    else:
        bg = "#a24ff020"  # purple tint for Both

    # use a small frame per entry
    row = customtkinter.CTkFrame(master=hist_scroll, fg_color=None, corner_radius=8)
    # left label: timestamp & filename
    left_txt = f"{entry['time']} - {os.path.basename(entry['file'])}"
    left_label = customtkinter.CTkLabel(master=row, text=left_txt, anchor="w")
    left_label.grid(row=0, column=0, padx=6, pady=6, sticky="w")
    # middle label: model and label
    mid_txt = f"{entry['model']} | {entry['label']}"
    mid_label = customtkinter.CTkLabel(master=row, text=mid_txt)
    mid_label.grid(row=0, column=1, padx=6, pady=6)
    # right label: probs
    right_txt = f"real:{entry['prob_real']:.3f} fake:{entry['prob_fake']:.3f}"
    right_label = customtkinter.CTkLabel(master=row, text=right_txt)
    right_label.grid(row=0, column=2, padx=6, pady=6)

    # configure background by applying fg_color on a nested frame
    color_frame = customtkinter.CTkFrame(master=row, fg_color=None)
    # pack row
    row.pack(fill="x", padx=4, pady=4)
    # set left/mid/right weights
    row.grid_columnconfigure(0, weight=3)
    row.grid_columnconfigure(1, weight=1)
    row.grid_columnconfigure(2, weight=1)

    # set style: since CTk widgets don't let us set background for a single label easily,
    # color the labels by changing text color for contrast (we'll keep subtle backgrounds minimal).
    # Use different text colors for model types:
    if entry['model'] == "RandomForest":
        left_label.configure(text_color="#A9D1FF")
        mid_label.configure(text_color="#D7E9FF")
        right_label.configure(text_color="#D7E9FF")
    elif entry['model'] == "CNN":
        left_label.configure(text_color="#BFF5C1")
        mid_label.configure(text_color="#DFFFE8")
        right_label.configure(text_color="#DFFFE8")
    else:
        left_label.configure(text_color="#E9D7FF")
        mid_label.configure(text_color="#F5E8FF")
        right_label.configure(text_color="#F5E8FF")

def append_history(entry):
    """
    entry is expected to be a dict with keys:
      'time', 'file', 'label', 'prob_real', 'prob_fake', 'model'
    """
    HISTORY.append(entry)
    _add_history_row_widget(entry)

def save_history_csv():
    if not HISTORY:
        status_label.configure(text="No history to save.")
        return
    fname = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
    if not fname:
        return
    try:
        with open(fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","file","model","label","prob_real","prob_fake"])
            for e in HISTORY:
                w.writerow([e["time"], e["file"], e["model"], e["label"], e["prob_real"], e["prob_fake"]])
        status_label.configure(text=f"History saved: {fname}")
    except Exception as e:
        status_label.configure(text=f"Failed to save: {e}")

def clear_history():
    global HISTORY
    HISTORY = []
    # destroy all children of hist_scroll
    for child in hist_scroll.winfo_children():
        child.destroy()
    status_label.configure(text="History cleared")

# helper: plot waveform & mel on given axes
def plot_wave_and_mel(y, sr, ax_wf, ax_mel, title_prefix=""):
    try:
        ax_wf.clear(); ax_mel.clear()
        times = np.arange(len(y)) / sr
        ax_wf.plot(times, y, linewidth=0.6)
        ax_wf.set_title(f"{title_prefix} Waveform")
        import librosa
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        ax_mel.imshow(S_db, origin="lower", aspect="auto")
        ax_mel.set_title(f"{title_prefix} Mel-spectrogram (dB)")
    except Exception as e:
        print("plot error", e)

# prediction logic
def do_predict(path, thresh, prefer_model=None):
    """
    predict on single model (RF or CNN) depending on model_choice_var or prefer_model param.
    prefer_model can be "RandomForest" or "CNN" to force choice.
    """
    model_used = "RandomForest"  # default
    try:
        choice = model_choice_var.get() if prefer_model is None else prefer_model
        # resolve actual model to use
        rf_model, rf_scaler = model_cache.get("model"), model_cache.get("scaler")
        cnn_model, cnn_meta = load_cnn_model()

        if choice == "RandomForest":
            use_cnn = False
            if rf_model is None:
                status_label.configure(text="RF model not found. Train RF or select CNN.")
                return
        elif choice == "CNN":
            use_cnn = True
            if cnn_model is None:
                status_label.configure(text="CNN model not found. Train CNN or select RF.")
                return
        else:  # Auto
            # prefer RF, else CNN
            if rf_model is not None:
                use_cnn = False
            elif cnn_model is not None:
                use_cnn = True
            else:
                status_label.configure(text="No model found (RF or CNN). Train a model first.")
                return

        if use_cnn:
            model_used = "CNN"
            try:
                res = predict_cnn_file(path, model=cnn_model, meta=cnn_meta) if predict_cnn_file is not None else None
            except Exception as e:
                status_label.configure(text=f"CNN prediction failed: {e}")
                traceback.print_exc()
                return
            if res is None:
                status_label.configure(text="CNN predict returned no result.")
                return
            prob_real = float(res.get("prob_real", 0.0))
            prob_fake = float(res.get("prob_fake", 1.0 - prob_real))
            label = "Fake" if prob_fake >= thresh else "Real"
        else:
            model_used = "RandomForest"
            model, scaler = rf_model, rf_scaler
            if model is None:
                status_label.configure(text="RF model not loaded. Reload or train.")
                return

            feats = extract_features_from_file(path)
            if feats is None:
                status_label.configure(text="Feature extraction failed.")
                return

            X = feats.reshape(1, -1)
            if scaler is not None:
                try:
                    X = scaler.transform(X)
                except Exception:
                    pass

            # compute probabilities robustly
            if hasattr(model, "predict_proba"):
                proba_arr = model.predict_proba(X)
                classes = list(getattr(model, "classes_", []))
                if 0 in classes and 1 in classes:
                    idx_fake = classes.index(0)
                    idx_real = classes.index(1)
                    prob_fake = float(proba_arr[0, idx_fake])
                    prob_real = float(proba_arr[0, idx_real])
                else:
                    try:
                        prob_real = float(proba_arr[0, 1])
                        prob_fake = 1.0 - prob_real
                    except Exception:
                        prob_fake = float(proba_arr[0, 0])
                        prob_real = 1.0 - prob_fake
            else:
                pred_val = float(model.predict(X)[0])
                prob_real = 1.0 if pred_val == 1 else 0.0
                prob_fake = 1.0 - prob_real

            label = "Fake" if prob_fake >= thresh else "Real"

        # update UI: labels
        result_label_var.set(f"{model_used}: {label}")
        prob_label_var.set(f"Probability (real): {prob_real:.4f}  —  (fake): {prob_fake:.4f}  —  Thr: {thresh:.2f}")
        if label == "Fake":
            result_label.configure(text_color="#FF6B6B")
        else:
            result_label.configure(text_color="#7BE495")

        # plot single-model visuals in the left column (RF -> left; CNN-> right)
        # For consistency, plot the same audio in both columns but annotate which model is displayed
        y_sr = safe_read_audio(path, target_sr=None, mono=True)
        if y_sr is not None:
            y, sr = y_sr
            # if model_used == RF: show RF visuals on left; else show CNN visuals on right
            if model_used == "RandomForest":
                plot_wave_and_mel(y, sr, ax_rf_wf, ax_rf_mel, title_prefix="RF")
                ax_cnn_wf.clear(); ax_cnn_mel.clear()
            else:
                plot_wave_and_mel(y, sr, ax_cnn_wf, ax_cnn_mel, title_prefix="CNN")
                ax_rf_wf.clear(); ax_rf_mel.clear()
            fig.tight_layout(); canvas.draw()
        else:
            ax_rf_wf.clear(); ax_rf_mel.clear(); ax_cnn_wf.clear(); ax_cnn_mel.clear(); canvas.draw()

        # append history
        entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": path,
            "label": label,
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "model": model_used
        }
        append_history(entry)
        status_label.configure(text=f"Predicted: {label} (prob_fake={prob_fake:.3f}) using {model_used}")
    except Exception as e:
        status_label.configure(text=f"Prediction failed: {e}")
        traceback.print_exc()

def run_predict_background():
    path = entry_file.get().strip()
    if not path or not os.path.exists(path):
        status_label.configure(text="No file selected or path invalid.")
        return
    # respect explicit model_choice selection (Auto/RandomForest/CNN)
    thresh = threshold_var.get()
    status_label.configure(text="Predicting...")
    threading.Thread(target=lambda: do_predict(path, thresh), daemon=True).start()

btn_run.configure(command=run_predict_background)

# --- Run Both (RF + CNN) ---
def run_both(path, thresh):
    """
    Run both RF and CNN predictions, plot visuals side-by-side and show comparison.
    (Replacement function — sets the main prediction label & prob line as well.)
    """
    try:
        rf_model, rf_scaler = model_cache.get("model"), model_cache.get("scaler")
        cnn_model, cnn_meta = load_cnn_model()

        if rf_model is None and cnn_model is None:
            status_label.configure(text="No RF or CNN model available. Train one or both.")
            return

        # helper to format probabilities safely
        def fmt(p):
            return "N/A" if p is None else f"{p:.4f}"

        # RF prediction
        rf_prob_real = rf_prob_fake = None
        if rf_model is not None:
            feats = extract_features_from_file(path)
            if feats is None:
                status_label.configure(text="Feature extraction failed for RF.")
                return
            X = feats.reshape(1, -1)
            if rf_scaler is not None:
                try:
                    X = rf_scaler.transform(X)
                except Exception:
                    pass
            if hasattr(rf_model, "predict_proba"):
                proba_arr = rf_model.predict_proba(X)
                classes = list(getattr(rf_model, "classes_", []))
                if 0 in classes and 1 in classes:
                    idx_fake = classes.index(0)
                    idx_real = classes.index(1)
                    rf_prob_fake = float(proba_arr[0, idx_fake])
                    rf_prob_real = float(proba_arr[0, idx_real])
                else:
                    try:
                        rf_prob_real = float(proba_arr[0, 1])
                        rf_prob_fake = 1.0 - rf_prob_real
                    except Exception:
                        rf_prob_fake = float(proba_arr[0, 0])
                        rf_prob_real = 1.0 - rf_prob_fake
            else:
                pred_val = float(rf_model.predict(X)[0])
                rf_prob_real = 1.0 if pred_val == 1 else 0.0
                rf_prob_fake = 1.0 - rf_prob_real

            rf_label = "Fake" if rf_prob_fake >= thresh else "Real"
        else:
            rf_label = "N/A"

        # CNN prediction
        cnn_prob_real = cnn_prob_fake = None
        if cnn_model is not None:
            try:
                res = predict_cnn_file(path, model=cnn_model, meta=cnn_meta)
                cnn_prob_real = float(res.get("prob_real", 0.0))
                cnn_prob_fake = float(res.get("prob_fake", 1.0 - cnn_prob_real))
                cnn_label = "Fake" if cnn_prob_fake >= thresh else "Real"
            except Exception as e:
                status_label.configure(text=f"CNN prediction failed: {e}")
                traceback.print_exc()
                cnn_label = "Err"
        else:
            cnn_label = "N/A"

        # plot both visuals: use same audio for both columns, labelled RF/CNN
        y_sr = safe_read_audio(path, target_sr=None, mono=True)
        if y_sr is not None:
            y, sr = y_sr
            if rf_model is not None:
                plot_wave_and_mel(y, sr, ax_rf_wf, ax_rf_mel, title_prefix="RF")
            else:
                ax_rf_wf.clear(); ax_rf_mel.clear()
            if cnn_model is not None:
                plot_wave_and_mel(y, sr, ax_cnn_wf, ax_cnn_mel, title_prefix="CNN")
            else:
                ax_cnn_wf.clear(); ax_cnn_mel.clear()
            fig.tight_layout(); canvas.draw()
        else:
            ax_rf_wf.clear(); ax_rf_mel.clear(); ax_cnn_wf.clear(); ax_cnn_mel.clear(); canvas.draw()

        # comparison summary string
        comp_lines = []
        comp_lines.append("Comparison (RF vs CNN):")
        comp_lines.append(f"RF -> real: {fmt(rf_prob_real)} fake: {fmt(rf_prob_fake)} -> {rf_label}" if rf_prob_real is not None else "RF -> N/A")
        comp_lines.append(f"CNN -> real: {fmt(cnn_prob_real)} fake: {fmt(cnn_prob_fake)} -> {cnn_label}" if cnn_prob_real is not None else "CNN -> N/A")
        if rf_prob_fake is not None and cnn_prob_fake is not None:
            agree = (rf_label == cnn_label)
            comp_lines.append(f"Agreement: {'YES' if agree else 'NO'}")
            comp_lines.append(f"Delta (prob_fake RF - prob_fake CNN): { (rf_prob_fake - cnn_prob_fake):+0.4f}")
        comparison_var.set("\n".join(comp_lines))

        # build averaged probs for history / main display
        if rf_prob_real is not None and cnn_prob_real is not None:
            avg_prob_real = (rf_prob_real + cnn_prob_real) / 2.0
            avg_prob_fake = 1.0 - avg_prob_real
        elif rf_prob_real is not None:
            avg_prob_real = rf_prob_real; avg_prob_fake = rf_prob_fake
        elif cnn_prob_real is not None:
            avg_prob_real = cnn_prob_real; avg_prob_fake = cnn_prob_fake
        else:
            avg_prob_real = None; avg_prob_fake = None

        # decide final label by simple majority: if both present and disagree, mark 'Conflict'
        if rf_label != "N/A" and cnn_label != "N/A" and rf_label != "Err" and cnn_label != "Err":
            if rf_label == cnn_label:
                final_label = rf_label
            else:
                final_label = f"Conflict ({rf_label}/{cnn_label})"
        else:
            # choose whichever is available (prefer RF if present)
            if rf_label not in ("N/A", "Err"):
                final_label = rf_label
            elif cnn_label not in ("N/A", "Err"):
                final_label = cnn_label
            else:
                final_label = "N/A"

        # --- NEW: update main result label & prob line ---
        # Show final decision and the three probability summaries
        result_text = f"Both: {final_label}"
        result_label_var.set(result_text)

        prob_text_parts = [
            f"RF real: {fmt(rf_prob_real)} fake: {fmt(rf_prob_fake)}",
            f"CNN real: {fmt(cnn_prob_real)} fake: {fmt(cnn_prob_fake)}"
        ]
        if avg_prob_real is not None:
            prob_text_parts.append(f"Avg real: {avg_prob_real:.4f} fake: {avg_prob_fake:.4f}")
        prob_label_var.set("  |  ".join(prob_text_parts))

        # color coding: Fake -> red, Real -> green, Conflict -> orange, N/A -> neutral
        if "Fake" in final_label and "Conflict" not in final_label:
            result_label.configure(text_color="#FF6B6B")
        elif "Real" in final_label and "Conflict" not in final_label:
            result_label.configure(text_color="#7BE495")
        elif "Conflict" in final_label:
            result_label.configure(text_color="#FFA500")  # orange
        else:
            result_label.configure(text_color="#FFFFFF")  # default/white

        # append single history row marking 'Both'
        entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": path,
            "label": final_label,
            "prob_real": avg_prob_real if avg_prob_real is not None else 0.0,
            "prob_fake": avg_prob_fake if avg_prob_fake is not None else 1.0,
            "model": "Both"
        }
        append_history(entry)
        status_label.configure(text=f"Run Both complete. RF:{rf_label} CNN:{cnn_label}")
    except Exception as e:
        status_label.configure(text=f"Run Both failed: {e}")
        traceback.print_exc()

def run_both_background():
    path = entry_file.get().strip()
    if not path or not os.path.exists(path):
        status_label.configure(text="No file selected or path invalid.")
        return
    thresh = threshold_var.get()
    status_label.configure(text="Running both models...")
    threading.Thread(target=lambda: run_both(path, thresh), daemon=True).start()

btn_run_both.configure(command=run_both_background)

# train background (RF)
def run_train_background():
    status_label.configure(text="Training RF started... see console for logs.")
    def task():
        try:
            p = train()  # call RF train from train.py; it will save RF model at MODEL_PATH
            time.sleep(0.8)
            # reset cnn cache (training CNN not handled here)
            cnn_cache["model"], cnn_cache["meta"], cnn_cache["loaded"] = None, None, False
            refresh_model_info()
            status_label.configure(text=f"RF Training finished: {p}")
        except Exception as e:
            status_label.configure(text=f"Training failed: {e}")
            traceback.print_exc()
    threading.Thread(target=task, daemon=True).start()

btn_train.configure(command=run_train_background)

# initial model info refresh
refresh_model_info()

# start app
app.mainloop()
