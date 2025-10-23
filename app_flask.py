# app_flask.py
import os
import io
import csv
import base64
import tempfile
import traceback
from datetime import datetime
from functools import lru_cache

from flask import Flask, request, render_template_string, redirect, url_for, send_file, flash
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import your helpers (features.py, cnn_predict.py, train.py)
try:
    from features import safe_read_audio, extract_features_from_file
except Exception:
    safe_read_audio = None
    extract_features_from_file = None

try:
    from cnn_predict import load_cnn as load_cnn_fn, predict_file as predict_cnn_file
    CNN_AVAILABLE = True
except Exception:
    load_cnn_fn = None
    predict_cnn_file = None
    CNN_AVAILABLE = False

# DEFAULT MODEL PATH fallback
try:
    from train import DEFAULT_MODEL_PATH as DEFAULT_RF_PATH
except Exception:
    DEFAULT_RF_PATH = os.path.join("models", "rf_model.joblib")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me-for-production")

# Simple in-memory history (list of dicts)
HISTORY = []

# ----------------------
# Helpers: model loading
# ----------------------
@lru_cache(maxsize=1)
def load_rf_model(path=DEFAULT_RF_PATH):
    """Return (model, scaler) or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        data = joblib.load(path)
        if isinstance(data, dict) and "model" in data:
            model = data.get("model")
            scaler = data.get("scaler")
        else:
            model = data
            scaler = None
        return model, scaler
    except Exception as e:
        print("Failed to load RF model:", e)
        return None, None

@lru_cache(maxsize=1)
def load_cnn_model_cached():
    """Return (model, meta) if cnn_predict is available and loading succeeds."""
    if not CNN_AVAILABLE or load_cnn_fn is None:
        return None, None
    try:
        return load_cnn_fn()
    except Exception as e:
        print("Failed to load cnn model:", e)
        return None, None

# ----------------------
# Audio plotting helpers
# ----------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return img_b64

def make_wave_and_mel_png(y, sr, title_prefix="File"):
    """Return tuple (wave_b64, mel_b64) as base64 PNG strings."""
    try:
        # waveform
        fig1, ax1 = plt.subplots(figsize=(8,2.2))
        times = np.arange(len(y)) / float(sr) if sr and len(y) else np.array([0])
        ax1.plot(times, y, linewidth=0.6)
        ax1.set_title(f"{title_prefix} — Waveform")
        ax1.set_xlabel("Seconds")
        fig1.tight_layout()
        wave_b64 = fig_to_base64(fig1)

        # mel-spectrogram
        try:
            import librosa
            import librosa.display
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            fig2, ax2 = plt.subplots(figsize=(8,3.2))
            im = ax2.imshow(S_db, origin="lower", aspect="auto")
            ax2.set_title(f"{title_prefix} — Mel-spectrogram (dB)")
            ax2.set_xlabel("Frames")
            ax2.set_ylabel("Mel bins")
            fig2.tight_layout()
            mel_b64 = fig_to_base64(fig2)
        except Exception:
            mel_b64 = None

        return wave_b64, mel_b64
    except Exception as e:
        print("make_wave_and_mel_png error:", e)
        traceback.print_exc()
        return None, None

# ----------------------
# Prediction helpers
# ----------------------
def rf_predict_proba(model, scaler, feats):
    """Return (prob_real, prob_fake) robustly or (None, None)."""
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

# ----------------------
# Routes & UI
# ----------------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Deepfake Audio Detector — Web</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; background:#0f1720; color:#e6eef6; margin:0; padding:18px; }
    .container { max-width:1100px; margin:0 auto; }
    .card { background:#12151a; padding:14px; border-radius:10px; box-shadow:0 4px 14px rgba(0,0,0,0.6); }
    h1 { margin:0 0 12px 0; font-size:22px; }
    label { display:block; margin-top:8px; font-size:14px; color:#9aa3ad; }
    input[type=file] { color:#fff; }
    .row { display:flex; gap:12px; margin-top:12px; }
    .col { flex:1; }
    .small { font-size:13px; color:#9aa3ad; }
    .btn { background:#1f6feb; color:#fff; padding:8px 12px; border:none; border-radius:6px; cursor:pointer; }
    .btn.secondary { background:#444; }
    .history { max-height:240px; overflow:auto; margin-top:12px; }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:6px 8px; text-align:left; border-bottom:1px solid #222; font-size:13px; }
    .green{color:#7BE495} .red{color:#FF6B6B} .orange{color:#FFA500}
    img.resp { max-width:100%; border-radius:6px; border:1px solid #222; }
    footer { margin-top:20px; font-size:12px; color:#8a8a8a; text-align:center; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>🎙️ Deepfake Audio Detector — Web</h1>
      <div class="small">Upload an audio file, choose model(s) and run prediction. Developed by <strong>Alok Kushwaha</strong>.</div>

      <form method="post" action="{{ url_for('predict') }}" enctype="multipart/form-data">
        <label>Upload audio file</label>
        <input type="file" name="audio_file" accept=".wav,.mp3,.flac,.ogg,.m4a" required>

        <div class="row">
          <div class="col">
            <label>Model</label>
            <select name="model_choice">
              <option value="Auto" {% if model_choice=='Auto' %}selected{% endif %}>Auto (prefer RF)</option>
              <option value="RandomForest" {% if model_choice=='RandomForest' %}selected{% endif %}>RandomForest (MFCC)</option>
              <option value="CNN" {% if model_choice=='CNN' %}selected{% endif %}>CNN (spectrogram)</option>
              <option value="Both" {% if model_choice=='Both' %}selected{% endif %}>Run Both (RF + CNN)</option>
            </select>
            <label class="small">Decision threshold (prob_fake ≥ threshold → Fake)</label>
            <input type="range" min="0" max="1" step="0.01" name="threshold" value="{{ threshold|default(0.50) }}" oninput="this.nextElementSibling.value = this.value">
            <output>{{ threshold|default(0.50) }}</output>
          </div>

          <div class="col" style="display:flex; flex-direction:column; justify-content:flex-end;">
            <button class="btn" type="submit">▶️ Run Prediction</button>
            &nbsp;
            <a class="btn secondary" href="{{ url_for('download_history') }}">⬇️ Download History CSV</a>
          </div>
        </div>
      </form>
    </div>

    {% if result %}
    <div style="height:12px"></div>
    <div class="card">
      <h2>Result — <span class="{% if result.final_label=='Fake' %}red{% elif 'Conflict' in result.final_label %}orange{% else %}green{% endif %}">{{ result.final_label }}</span></h2>
      <p class="small">Model run: {{ result.model_used }} &nbsp; | &nbsp; File: {{ result.filename }} &nbsp; | &nbsp; Time: {{ result.time }}</p>

      <div style="display:flex; gap:12px; margin-top:8px;">
        <div style="flex:1;">
          <h4 class="small">RF</h4>
          <div class="small">real: {{ result.rf_prob_real }} — fake: {{ result.rf_prob_fake }}</div>
        </div>
        <div style="flex:1;">
          <h4 class="small">CNN</h4>
          <div class="small">real: {{ result.cnn_prob_real }} — fake: {{ result.cnn_prob_fake }}</div>
        </div>
        <div style="flex:1;">
          <h4 class="small">Average</h4>
          <div class="small">real: {{ result.avg_real }} — fake: {{ result.avg_fake }}</div>
        </div>
      </div>

      <div style="display:flex; gap:12px; margin-top:12px;">
        <div style="flex:1">
          <h4 class="small">Waveform</h4>
          {% if result.wave_b64 %}
            <img class="resp" src="data:image/png;base64,{{ result.wave_b64 }}" alt="waveform">
          {% else %}
            <div class="small">Waveform not available.</div>
          {% endif %}
        </div>
        <div style="flex:1">
          <h4 class="small">Mel-spectrogram</h4>
          {% if result.mel_b64 %}
            <img class="resp" src="data:image/png;base64,{{ result.mel_b64 }}" alt="mel">
          {% else %}
            <div class="small">Mel-spectrogram not available.</div>
          {% endif %}
        </div>
      </div>
    </div>
    {% endif %}

    <div style="height:12px"></div>
    <div class="card">
      <h3>Prediction history (session)</h3>
      <div class="history">
        {% if history %}
          <table>
            <thead><tr><th>Time</th><th>File</th><th>Model</th><th>Label</th><th>prob_real</th><th>prob_fake</th></tr></thead>
            <tbody>
              {% for h in history|reverse %}
              <tr>
                <td>{{ h.time }}</td>
                <td>{{ h.file }}</td>
                <td>{{ h.model }}</td>
                <td class="{% if 'Fake' in h.label and 'Conflict' not in h.label %}red{% elif 'Conflict' in h.label %}orange{% else %}green{% endif %}">{{ h.label }}</td>
                <td>{{ h.prob_real }}</td>
                <td>{{ h.prob_fake }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        {% else %}
          <div class="small">No predictions this session yet.</div>
        {% endif %}
      </div>
    </div>

    <footer>© {{ year }} — Developed by Alok Kushwaha</footer>
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    # default values
    return render_template_string(INDEX_HTML,
                                  result=None,
                                  history=HISTORY,
                                  year=datetime.now().year,
                                  model_choice="Auto",
                                  threshold=0.50)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "audio_file" not in request.files:
            flash("No file part", "error")
            return redirect(url_for("index"))

        f = request.files["audio_file"]
        if f.filename == "":
            flash("No file selected", "error")
            return redirect(url_for("index"))

        # save to temp file
        suffix = os.path.splitext(f.filename)[1] or ".wav"
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.save(tf.name)
        tf.flush(); tf.close()
        filepath = tf.name

        # inputs
        model_choice = request.form.get("model_choice", "Auto")
        thresh = float(request.form.get("threshold", 0.50))

        # load models
        rf_model, rf_scaler = load_rf_model()
        cnn_model, cnn_meta = load_cnn_model_cached()

        # prepare result containers
        rf_prob_real = rf_prob_fake = None
        cnn_prob_real = cnn_prob_fake = None
        model_used = None

        # pick which to run
        choice = model_choice
        if choice == "Auto":
            if rf_model is not None:
                choice = "RandomForest"
            elif cnn_model is not None:
                choice = "CNN"

        # RF
        if choice in ("RandomForest", "Both"):
            if rf_model is not None:
                feats = extract_features_from_file(filepath) if extract_features_from_file is not None else None
                if feats is None:
                    rf_prob_real, rf_prob_fake = None, None
                else:
                    rf_prob_real, rf_prob_fake = rf_predict_proba(rf_model, rf_scaler, feats)
            else:
                rf_prob_real, rf_prob_fake = None, None

        # CNN
        if choice in ("CNN", "Both") or choice == "Both":
            if cnn_model is not None and predict_cnn_file is not None:
                try:
                    res = predict_cnn_file(filepath, model=cnn_model, meta=cnn_meta)
                    cnn_prob_real = float(res.get("prob_real", 0.0))
                    cnn_prob_fake = float(res.get("prob_fake", 1.0 - cnn_prob_real))
                except Exception as e:
                    print("cnn predict error:", e)
                    cnn_prob_real, cnn_prob_fake = None, None
            else:
                cnn_prob_real, cnn_prob_fake = None, None

        # Decide final label and model_used
        # prefer RF if selected single; if Both compute average
        final_label = "N/A"
        avg_real = avg_fake = None

        if choice == "RandomForest":
            model_used = "RandomForest"
            if rf_prob_real is None:
                final_label = "N/A"
            else:
                final_label = "Fake" if rf_prob_fake >= thresh else "Real"
                avg_real, avg_fake = rf_prob_real, rf_prob_fake

        elif choice == "CNN":
            model_used = "CNN"
            if cnn_prob_real is None:
                final_label = "N/A"
            else:
                final_label = "Fake" if cnn_prob_fake >= thresh else "Real"
                avg_real, avg_fake = cnn_prob_real, cnn_prob_fake

        elif choice == "Both":
            model_used = "Both"
            # compute average of reals if available, else whichever exists
            reals = []
            fakes = []
            if rf_prob_real is not None:
                reals.append(rf_prob_real); fakes.append(rf_prob_fake)
            if cnn_prob_real is not None:
                reals.append(cnn_prob_real); fakes.append(cnn_prob_fake)
            if reals:
                avg_real = float(sum(reals) / len(reals))
                avg_fake = float(sum(fakes) / len(fakes))
                # labels per model
                rf_label = None if rf_prob_fake is None else ("Fake" if rf_prob_fake >= thresh else "Real")
                cnn_label = None if cnn_prob_fake is None else ("Fake" if cnn_prob_fake >= thresh else "Real")
                if rf_label and cnn_label and rf_label == cnn_label:
                    final_label = rf_label
                elif rf_label and cnn_label and rf_label != cnn_label:
                    final_label = f"Conflict ({rf_label}/{cnn_label})"
                else:
                    final_label = rf_label or cnn_label or "N/A"
            else:
                avg_real = avg_fake = None
                final_label = "N/A"
        else:
            model_used = choice
            final_label = "N/A"

        # Load audio visuals (wave + mel)
        wave_b64 = mel_b64 = None
        try:
            if safe_read_audio is not None:
                y_sr = safe_read_audio(filepath, target_sr=None, mono=True)
                if y_sr is not None:
                    y, sr = y_sr
                    wave_b64, mel_b64 = make_wave_and_mel_png(y, sr, title_prefix="File")
            else:
                # try librosa fallback
                import librosa
                y, sr = librosa.load(filepath, sr=None, mono=True)
                wave_b64, mel_b64 = make_wave_and_mel_png(y, sr, title_prefix="File")
        except Exception as e:
            print("visual generation failed:", e)
            wave_b64 = mel_b64 = None

        # build result object to render
        result = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": os.path.basename(filepath),
            "model_used": model_used,
            "final_label": final_label,
            "rf_prob_real": rf_prob_real if rf_prob_real is not None else "N/A",
            "rf_prob_fake": rf_prob_fake if rf_prob_fake is not None else "N/A",
            "cnn_prob_real": cnn_prob_real if cnn_prob_real is not None else "N/A",
            "cnn_prob_fake": cnn_prob_fake if cnn_prob_fake is not None else "N/A",
            "avg_real": avg_real if avg_real is not None else "N/A",
            "avg_fake": avg_fake if avg_fake is not None else "N/A",
            "wave_b64": wave_b64,
            "mel_b64": mel_b64
        }

        # append to HISTORY (store numeric floats or N/A)
        history_item = {
            "time": result["time"],
            "file": result["filename"],
            "model": model_used,
            "label": final_label,
            "prob_real": (avg_real if isinstance(avg_real, float) else (rf_prob_real if isinstance(rf_prob_real, float) else (cnn_prob_real if isinstance(cnn_prob_real, float) else None))),
            "prob_fake": (avg_fake if isinstance(avg_fake, float) else (rf_prob_fake if isinstance(rf_prob_fake, float) else (cnn_prob_fake if isinstance(cnn_prob_fake, float) else None)))
        }
        HISTORY.append(history_item)

        # render page with result and updated history
        return render_template_string(INDEX_HTML,
                                      result=result,
                                      history=HISTORY,
                                      year=datetime.now().year,
                                      model_choice=model_choice,
                                      threshold=thresh)
    except Exception as e:
        print("Predict failed:", e)
        traceback.print_exc()
        flash(f"Prediction failed: {e}", "error")
        return redirect(url_for("index"))

@app.route("/download_history")
def download_history():
    # make CSV in-memory
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["timestamp","file","model","label","prob_real","prob_fake"])
    for h in HISTORY:
        cw.writerow([h.get("time"), h.get("file"), h.get("model"), h.get("label"), h.get("prob_real"), h.get("prob_fake")])
    mem = io.BytesIO()
    mem.write(si.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="prediction_history.csv")

# ----------------------
# CLI helpers & run
# ----------------------
if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
