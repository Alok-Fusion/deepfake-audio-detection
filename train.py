# train.py — improved training script
import os
import json
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from features import extract_features_from_file, list_files_labels

MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
IMPORTANCE_CSV = os.path.join(MODEL_DIR, "feature_importances.csv")

def parse_args():
    p = argparse.ArgumentParser(description="Train RandomForest audio fake/real classifier")
    p.add_argument("--data-dir", default="data", help="Root data folder containing 'real' and 'fake' subfolders")
    p.add_argument("--n-estimators", type=int, default=200, help="Number of trees for RandomForest")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                   help="Use class_weight='balanced' or no class weight ('none')")
    p.add_argument("--cross-val", action="store_true", help="Run 5-fold cross validation after training")
    p.add_argument("--max-files", type=int, default=0, help="(Optional) limit number of files processed (0 = all)")
    return p.parse_args()

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def train(data_dir="data", n_estimators=200, test_size=0.2, random_state=42,
          class_weight="balanced", do_cross_val=False, max_files=0):
    files, labels = list_files_labels(data_dir)
    if not files:
        raise RuntimeError(f"No audio files found under {data_dir}/real and {data_dir}/fake")

    if max_files and max_files > 0:
        files = files[:max_files]
        labels = labels[:max_files]

    print(f"[{datetime.now().isoformat()}] Found {len(files)} files. Extracting features...")

    X = []
    y = []
    failed = []
    for f, lab in tqdm(zip(files, labels), total=len(files), unit="file"):
        try:
            feat = extract_features_from_file(f)
            if feat is None:
                failed.append(f)
                continue
            X.append(feat)
            y.append(lab)
        except Exception as e:
            failed.append(f)
            print(f"  Error extracting {f}: {e}")

    print(f"[{datetime.now().isoformat()}] Extracted features for {len(X)} files; failed for {len(failed)} files.")
    if len(X) == 0:
        raise RuntimeError("Feature extraction failed for all files — aborting.")

    X = np.vstack(X)
    y = np.array(y)

    # Scale features
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)
    cw = None if class_weight == "none" else "balanced"
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                 random_state=random_state, class_weight=cw)

    print(f"[{datetime.now().isoformat()}] Training RandomForest (n_estimators={n_estimators}, class_weight={cw})...")
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=4, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "n_files_total": len(files),
        "n_features": X.shape[1],
        "test_size": test_size,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "class_weight": class_weight,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "failed_files_count": len(failed),
        "failed_files_sample": failed[:20]
    }

    print(f"[{datetime.now().isoformat()}] Test accuracy: {acc:.4f}")
    print("Classification report (text):")
    print(classification_report(y_test, preds, digits=4))

    # Optional cross-val
    if do_cross_val:
        print("[%s] Running 5-fold cross-validation (accuracy)..." % datetime.now().isoformat())
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        try:
            scores = cross_val_score(clf, Xs, y, cv=skf, scoring='accuracy', n_jobs=-1)
            metrics["cross_val_accuracy_mean"] = float(np.mean(scores))
            metrics["cross_val_accuracy_std"] = float(np.std(scores))
            print(f"Cross-val accuracy mean: {np.mean(scores):.4f} std: {np.std(scores):.4f}")
        except Exception as e:
            print("Cross-val failed:", e)

    # Save model + scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler}, DEFAULT_MODEL_PATH, compress=3)
    print(f"[{datetime.now().isoformat()}] Saved model to {DEFAULT_MODEL_PATH}")

    # Feature importances (if RF)
    try:
        importances = clf.feature_importances_
        # Save top-k importances
        idx = np.argsort(importances)[::-1]
        with open(IMPORTANCE_CSV, "w", encoding="utf-8") as fh:
            fh.write("rank,feature_index,importance\n")
            for rank, i in enumerate(idx, start=1):
                fh.write(f"{rank},{int(i)},{float(importances[i])}\n")
        metrics["feature_importances_csv"] = IMPORTANCE_CSV
        print(f"[{datetime.now().isoformat()}] Saved feature importances to {IMPORTANCE_CSV}")
    except Exception as e:
        print("Failed to compute/save feature importances:", e)

    # Save metrics JSON
    save_json(metrics, METRICS_PATH)
    print(f"[{datetime.now().isoformat()}] Saved metrics to {METRICS_PATH}")

    return DEFAULT_MODEL_PATH, metrics

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def main():
    args = parse_args()
    try:
        model_path, metrics = train(data_dir=args.data_dir,
                         n_estimators=args.n_estimators,
                         test_size=args.test_size,
                         random_state=args.random_state,
                         class_weight=args.class_weight,
                         do_cross_val=args.cross_val,
                         max_files=args.max_files)
    except Exception as e:
        print("Training failed:", e)
        raise

if __name__ == "__main__":
    main()
