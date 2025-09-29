# cnn_train.py
"""
Build and train a simple CNN on log-mel spectrograms produced by cnn_data.AudioSequence.
Saves best model to models/cnn_audio_fake_detector.h5 and metadata to models/cnn_meta.joblib

This version avoids passing `workers` / `use_multiprocessing` into model.fit()
to maintain compatibility across TF/Keras versions.
"""
import os
import joblib
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from features import list_files_labels   # reuse file listing (data/real, data/fake)
from cnn_data import AudioSequence, N_MELS, TIME_STEPS, SR

MODEL_DIR = "models"
CNN_BEST_PATH = os.path.join(MODEL_DIR, "cnn_audio_fake_detector.h5")
CNN_FINAL_PATH = os.path.join(MODEL_DIR, "cnn_audio_fake_detector_final.h5")
META_PATH = os.path.join(MODEL_DIR, "cnn_meta.joblib")

def build_cnn(input_shape=(N_MELS, TIME_STEPS, 1)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation='sigmoid')(x)  # prob of class 1 (real)
    model = models.Model(inputs=inp, outputs=out)
    return model

def train(data_dir="data", epochs=20, batch_size=16, lr=1e-3):
    files, labels = list_files_labels(data_dir)
    if not files:
        raise RuntimeError("No files found under data/real and data/fake")

    # split data: train / val / test
    f_train, f_temp, y_train, y_temp = train_test_split(files, labels, test_size=0.30, random_state=42, stratify=labels)
    f_val, f_test, y_val, y_test = train_test_split(f_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print("Train/Val/Test sizes:", len(f_train), len(f_val), len(f_test))
    print(f"Using batch_size={batch_size}, epochs={epochs}, lr={lr}")

    train_seq = AudioSequence(f_train, y_train, batch_size=batch_size, shuffle=True, augment=True)
    val_seq = AudioSequence(f_val, y_val, batch_size=batch_size, shuffle=False, augment=False)
    test_seq = AudioSequence(f_test, y_test, batch_size=batch_size, shuffle=False, augment=False)

    model = build_cnn(input_shape=(N_MELS, TIME_STEPS, 1))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt = callbacks.ModelCheckpoint(CNN_BEST_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1)
    es = callbacks.EarlyStopping(monitor='val_auc', patience=6, mode='max', restore_best_weights=True, verbose=1)
    rl = callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, verbose=1)
    csvlog = callbacks.CSVLogger(os.path.join(MODEL_DIR, 'cnn_training_log.csv'))

    print("Training start:", datetime.now().isoformat())
    try:
        # NOTE: do NOT pass `workers` or `use_multiprocessing` to keep compatibility
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=epochs,
            callbacks=[ckpt, es, rl, csvlog]
        )
    except Exception as e:
        print("model.fit raised an exception:", e)
        raise

    # Evaluate on test set
    eval_res = model.evaluate(test_seq, verbose=1)
    print("Test eval (loss, acc, auc):", eval_res)

    # Save final model & metadata
    model.save(CNN_FINAL_PATH)
    joblib.dump({"n_mels": N_MELS, "time_steps": TIME_STEPS, "sr": SR, "duration": 5.0}, META_PATH)
    print("Saved best model:", CNN_BEST_PATH)
    print("Saved final model:", CNN_FINAL_PATH)
    print("Saved metadata:", META_PATH)
    return CNN_BEST_PATH

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

def main():
    args = parse_args()
    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

if __name__ == "__main__":
    main()
