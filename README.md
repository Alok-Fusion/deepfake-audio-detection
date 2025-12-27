Sure — here’s a clean, Git-ready **README.md** you can drop into the repo. It’s compact, attractive, and contains all the Git-centric instructions you asked for (clone, branch, commit, PR, tags) plus setup, run, expected outputs and quick troubleshooting.

Copy the entire content below into `README.md` at your project root.

---

````markdown
# 🎙️ Audio Deepfake Detection

> Local desktop app to detect synthetic (fake) speech.  
> Uses MFCC + RandomForest for fast CPU inference, and an optional spectrogram CNN for higher accuracy. Includes a desktop GUI with single-file prediction, RF vs CNN comparison, and prediction history export.

---

**Tested:** Python **3.10.8**  
**Repo:** `audio-detection/`

---

## 🔎 What this repo contains
- `features.py` — audio loading and MFCC extraction (robust loader)
- `train.py` — RandomForest training pipeline (saves model + metrics)
- `cnn_train.py` — CNN training (spectrogram-based) — optional (requires TensorFlow)
- `cnn_predict.py` — CNN inference helper
- `ensemble_train.py` / `ensemble_predict.py` — optional stacking ensemble
- `gui_detailed.py` — desktop GUI (Predict, Run Both, visualization, history CSV)
- `requirements.txt` — pip dependencies
- `data/real`, `data/fake` — put your audio here
- `models/` — output models & metrics
- `images/ai.png` — optional icon

---

## ⚡ Quick setup (recommended)

Open a terminal at project root.

### 1. Clone
```bash
git clone <your-repo-url>
cd audio-detection
````

### 2. Create & activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> If installing `librosa` fails on Windows:
> `pip install numba llvmlite` then `pip install -r requirements.txt`.

### 4. (Optional) Install TensorFlow for CNN

```bash
pip install tensorflow
```

---

## ▶️ How to run

### Train RandomForest (MFCC features)

```bash
python train.py
```

Outputs:

* `models/rf_model.joblib` (model + scaler)
* `models/metrics.json`
* `models/feature_importances.csv`

### Train CNN (optional)

```bash
python cnn_train.py --epochs 20 --batch-size 16
```

Outputs:

* `models/cnn_audio_fake_detector.h5`
* `models/cnn_meta.joblib`

### Train Ensemble (optional)

```bash
python ensemble_train.py --rf models/rf_model.joblib --cnn models/cnn_audio_fake_detector.h5 --out models/ensemble_meta.joblib
```

### Run the GUI (desktop)

```bash
python gui_detailed.py
```

Use the GUI to:

* Browse → select file
* Model selector: Auto / RandomForest / CNN
* `Predict (Single)` or `Run Both` to compare RF vs CNN
* Adjust threshold, play audio (optional), save history CSV

### Deactivate venv

```bash
deactivate
```

---

## 📦 Expected artifacts & outputs

After training RF you should see in `models/`:

* `rf_model.joblib`
* `metrics.json` → contains `accuracy`, `confusion_matrix`, `classification_report`
* `feature_importances.csv`

Typical console sample:

```
Found 20000 files. Extracting features...
Training RandomForest...
Test accuracy: 0.683
Saved model to models/rf_model.joblib
Saved feature importances to models/feature_importances.csv
Saved metrics to models/metrics.json
```

In the GUI you will see:

* Waveform and Mel-spectrogram plots
* `prob_real` and `prob_fake` with threshold
* Colored history rows and CSV export

---

## 🩺 Short troubleshooting

* **mfcc() error / wrong imports** — ensure no local file is named `librosa.py` (it will shadow the real package).
* **librosa install fails (Windows)** — `pip install numba llvmlite` then `pip install -r requirements.txt`.
* **Keras/TensorFlow fit() workers error** — use the provided `cnn_train.py` (it avoids unsupported `fit` kwargs).
* **Audio load issues** — test:

  ```python
  from features import safe_read_audio
  y, sr = safe_read_audio('data/real/example.wav')
  print(y is not None, sr)
  ```
* **OneDrive file locks** — move project outside OneDrive if you see file-access errors.

---

## ✅ Git workflow (recommended)

### Basic commands

```bash
# initialize (if starting)
git init
git add .
git commit -m "Initial commit"

# clone an existing repo
git clone <repo-url>

# create and switch to a feature branch
git checkout -b feature/gui-improve

# stage & commit changes
git add .
git commit -m "Improve GUI: Run Both comparison and colored history"

# push branch and create PR
git push origin feature/gui-improve
```

Made with ❤️ — enjoy building and experimenting with local audio deepfake detection!

```

---

If you want the README as plain `readme.txt` instead (text-only), or a smaller one-page variant for a GitHub project front page, tell me which and I’ll generate it.
```
