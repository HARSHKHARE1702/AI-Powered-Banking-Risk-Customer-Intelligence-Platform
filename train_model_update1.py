
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

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
# 2. Load features
# -----------------------------------------

def load_features(name):

    npy_file = DATA_DIR / f"{name}.npy"
    npz_file = DATA_DIR / f"{name}.npz"

    if npy_file.exists():
        return np.load(npy_file)

    if npz_file.exists():
        return load_npz(npz_file)

    raise FileNotFoundError(name)


def load_labels(name):

    return pd.read_csv(
        DATA_DIR / f"{name}.csv"
    ).squeeze("columns").to_numpy()


X_train = load_features("X_train")
X_val = load_features("X_val")
X_test = load_features("X_test")

y_train = load_labels("y_train")
y_val = load_labels("y_val")
y_test = load_labels("y_test")


# -----------------------------------------
# 3. Validate datasets
# -----------------------------------------

datasets = [
    ("Training", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Testing", X_test, y_test)
]

for name, X, y in datasets:

    if X.shape[0] != len(y):
        raise ValueError(
            f"{name}: feature/label mismatch"
        )

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            f"{name}: invalid target values"
        )

if set(np.unique(y_train)) != {0, 1}:
    raise ValueError(
        "Training data must contain both classes"
    )

if set(np.unique(y_val)) != {0, 1}:
    raise ValueError(
        "Validation data must contain both classes"
    )

if set(np.unique(y_test)) != {0, 1}:
    raise ValueError(
        "Testing data must contain both classes"
    )


print("\n========== DATASET SUMMARY ==========")

for name, X, y in datasets:

    print(f"\n{name}")
    print("Features:", X.shape)
    print("Customers:", len(y))
    print("Default:", np.sum(y == 1))
    print("Non-default:", np.sum(y == 0))


# -----------------------------------------
# 4. Handle class imbalance
# -----------------------------------------

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = (
    negative_count / positive_count
)

print("\nScale positive weight:", scale_pos_weight)


# -----------------------------------------
# 5. Initialize XGBoost
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
# 7. Generate probabilities
# -----------------------------------------

train_prob = model.predict_proba(
    X_train
)[:, 1]

val_prob = model.predict_proba(
    X_val
)[:, 1]

test_prob = model.predict_proba(
    X_test
)[:, 1]


# -----------------------------------------
# 8. Select threshold using validation
# -----------------------------------------

thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = 0.50
best_f1 = -1

for threshold in thresholds:

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if current_f1 > best_f1:

        best_f1 = current_f1
        best_threshold = threshold


print("\n========== THRESHOLD SELECTION ==========")

print("Selected threshold:", best_threshold)
print("Validation F1:", best_f1)


# -----------------------------------------
# 9. Evaluation function
# -----------------------------------------

def evaluate_model(
    name,
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true, predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print(f"\n========== {name} RESULTS ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Non-default",
                "Default"
            ],
            zero_division=0
        )
    )

    return {
        "dataset": name,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# -----------------------------------------
# 10. Evaluate all three datasets
# -----------------------------------------

train_metrics = evaluate_model(
    "TRAINING",
    y_train,
    train_prob,
    best_threshold
)

val_metrics = evaluate_model(
    "VALIDATION",
    y_val,
    val_prob,
    best_threshold
)

test_metrics = evaluate_model(
    "TESTING",
    y_test,
    test_prob,
    best_threshold
)


# -----------------------------------------
# 11. Save evaluation results
# -----------------------------------------

metrics_df = pd.DataFrame([
    train_metrics,
    val_metrics,
    test_metrics
])

metrics_df.to_csv(
    ARTIFACT_DIR / "credit_model_metrics.csv",
    index=False
)


# -----------------------------------------
# 12. Save model and threshold
# -----------------------------------------

MODEL_PATH = (
    ARTIFACT_DIR / "credit_risk_xgb.joblib"
)

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold
    },
    MODEL_PATH
)

print("\nModel saved:", MODEL_PATH)

print(
    "\nEvaluation metrics saved successfully!"
)
