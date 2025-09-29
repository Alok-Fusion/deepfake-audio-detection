# gui.py
import os
import threading
import joblib
import customtkinter
from tkinter import filedialog
from PIL import Image
from features import extract_features_from_file, list_files_labels
from train import train, MODEL_PATH

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

app = customtkinter.CTk()
app.geometry("680x420")
app.title("Audio Fake Detector")

# Top frame (title)
frame_top = customtkinter.CTkFrame(master=app)
frame_top.pack(padx=16, pady=12, fill="x")
logo = None
if os.path.exists("images/ai.png"):
    logo = customtkinter.CTkImage(Image.open("images/ai.png"), size=(48,48))
title = customtkinter.CTkLabel(master=frame_top, text="  Deepfake Audio Detection", image=logo,
                              compound="left", font=customtkinter.CTkFont(size=18, weight="bold"))
title.pack(anchor="w")

# File select
frame_mid = customtkinter.CTkFrame(master=app)
frame_mid.pack(padx=16, pady=10, fill="x")
lbl = customtkinter.CTkLabel(master=frame_mid, text="Select audio file:")
lbl.grid(row=0, column=0, sticky="w", padx=6, pady=8)
entry = customtkinter.CTkEntry(master=frame_mid, width=420)
entry.grid(row=0, column=1, padx=6, pady=8)
def browse():
    p = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a")])
    if p:
        entry.delete(0, "end"); entry.insert(0, p); result_entry.delete(0, "end"); status_lbl.configure(text="")
browse_btn = customtkinter.CTkButton(master=frame_mid, text="Browse", command=browse)
browse_btn.grid(row=0, column=2, padx=6)

# Buttons + result
frame_ops = customtkinter.CTkFrame(master=app)
frame_ops.pack(padx=16, pady=8, fill="x")
result_entry = customtkinter.CTkEntry(master=frame_ops, width=420, fg_color="white", text_color="black", justify="center")
result_entry.grid(row=0, column=0, padx=8, pady=8)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    data = joblib.load(MODEL_PATH)
    model = data.get("model") if isinstance(data, dict) and 'model' in data else data
    scaler = data.get("scaler") if isinstance(data, dict) and 'scaler' in data else None
    return model, scaler

def predict(path):
    model, scaler = load_model()
    if model is None:
        return "Model not found. Train first (click Train)."
    feats = extract_features_from_file(path)
    if feats is None:
        return "Feature extraction failed."
    X = feats.reshape(1, -1)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    try:
        proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else float(model.predict(X)[0])
        label = "Fake" if proba >= 0.5 else "Real"
        return f"{label} (prob_fake={proba:.3f})"
    except Exception:
        return "Prediction error."

def on_run():
    path = entry.get().strip()
    if not path:
        result_entry.delete(0, "end"); result_entry.insert(0, "No file selected."); return
    result_entry.delete(0, "end"); result_entry.insert(0, "Processing..."); status_lbl.configure(text="Running...")
    def task():
        res = predict(path)
        result_entry.delete(0, "end"); result_entry.insert(0, res); status_lbl.configure(text="Done.")
    threading.Thread(target=task, daemon=True).start()

run_btn = customtkinter.CTkButton(master=frame_ops, text="Run", width=100, command=on_run)
run_btn.grid(row=0, column=1, padx=4)

def on_train():
    result_entry.delete(0, "end"); result_entry.insert(0, "Training..."); status_lbl.configure(text="Training...")
    def task_train():
        try:
            p = train()
            result_entry.delete(0, "end"); result_entry.insert(0, f"Trained -> {p}")
            status_lbl.configure(text="Training finished.")
        except Exception as e:
            result_entry.delete(0, "end"); result_entry.insert(0, f"Training failed: {e}")
            status_lbl.configure(text="Training failed.")
    threading.Thread(target=task_train, daemon=True).start()

train_btn = customtkinter.CTkButton(master=frame_ops, text="Train", width=100, command=on_train)
train_btn.grid(row=0, column=2, padx=4)

status_lbl = customtkinter.CTkLabel(master=app, text="")
status_lbl.pack(pady=(8,0))

footer = customtkinter.CTkLabel(master=app, text=f"Model path: {MODEL_PATH}", font=customtkinter.CTkFont(size=10))
footer.pack(side="bottom", pady=8)

app.mainloop()
