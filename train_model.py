from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz, issparse

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -----------------------------------------
# 1. Project paths
# -----------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Load processed features
# -----------------------------------------

def load_features(name):

    npy_file = DATA_DIR / f"{name}.npy"
    npz_file = DATA_DIR / f"{name}.npz"

    if npy_file.exists():
        return np.load(npy_file)

    if npz_file.exists():
        return load_npz(npz_file)

    raise FileNotFoundError(
        f"Could not find {npy_file} or {npz_file}"
    )


X_train = load_features("X_train")
X_test = load_features("X_test")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).squeeze("columns").to_numpy()

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze("columns").to_numpy()

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)


# -----------------------------------------
# 3. Validate data
# -----------------------------------------

if X_train.shape[0] != len(y_train):
    raise ValueError("Training feature/label mismatch")

if X_test.shape[0] != len(y_test):
    raise ValueError("Testing feature/label mismatch")

if set(np.unique(y_train)) != {0, 1}:
    raise ValueError("Training target must contain both 0 and 1")

if not set(np.unique(y_test)).issubset({0, 1}):
    raise ValueError("Testing target must contain only 0 and 1")


# -----------------------------------------
# 4. Handle class imbalance
# -----------------------------------------

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = (
    negative_count / positive_count
    if positive_count > 0 else 1.0
)

print("Non-default customers:", negative_count)
print("Default customers:", positive_count)
print("Scale positive weight:", scale_pos_weight)


# -----------------------------------------
# 5. Initialize XGBoost model
# -----------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# -----------------------------------------
# 6. Train model
# -----------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")


# -----------------------------------------
# 7. Generate predictions
# -----------------------------------------

y_pred = model.predict(X_test)

# Probability of class 1 (default)
y_probability = model.predict_proba(X_test)[:, 1]


# -----------------------------------------
# 8. Evaluate model
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, y_probability
)

print("\n========== CREDIT RISK RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-default", "Default"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------------
# 9. Save trained model
# -----------------------------------------

MODEL_PATH = ARTIFACT_DIR / "credit_risk_xgb.joblib"

joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)


# -----------------------------------------
# 10. Save evaluation metrics
# -----------------------------------------

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}

pd.DataFrame([metrics]).to_csv(
    ARTIFACT_DIR / "credit_model_metrics.csv",
    index=False
)

print("Evaluation metrics saved.")
