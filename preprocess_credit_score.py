from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------
# 1. File paths
# ---------------------------------

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "data" / "raw" / "customers.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# 2. Load dataset
# ---------------------------------

df = pd.read_csv(RAW_FILE)

print("Original dataset:", df.shape)

# ---------------------------------
# 3. Basic cleaning
# ---------------------------------

df = df.drop_duplicates()

TARGET = "default"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found")

df = df.dropna(subset=[TARGET])

# Convert target into binary labels.
# Adjust mappings if your generator uses different labels.

if df[TARGET].dtype == "object":
    target_mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    }

    df[TARGET] = (
        df[TARGET].astype(str).str.lower().str.strip()
        .map(target_mapping)
    )

if df[TARGET].isna().any():
    raise ValueError("Unrecognized or missing target labels")

df[TARGET] = df[TARGET].astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Expected binary target values 0 and 1")

# ---------------------------------
# 4. Separate features and target
# ---------------------------------

drop_columns = [
    TARGET,
    "customer_id"
]

X = df.drop(
    columns=[c for c in drop_columns if c in df.columns]
)

y = df[TARGET]

# ---------------------------------
# 5. Identify feature types
# ---------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ---------------------------------
# 6. Preprocessing pipelines
# ---------------------------------

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

# ---------------------------------
# 7. Train-test split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------
# 8. Fit only on training data
# ---------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# ---------------------------------
# 9. Save preprocessing artifacts
# ---------------------------------

joblib.dump(
    preprocessor,
    MODEL_DIR / "credit_preprocessor.joblib"
)

joblib.dump(
    list(X.columns),
    MODEL_DIR / "credit_feature_columns.joblib"
)

# Save transformed data in compressed sparse format
from scipy.sparse import save_npz, issparse
import numpy as np

# Save training features
if issparse(X_train_processed):
    save_npz(
        PROCESSED_DIR / "X_train.npz",
        X_train_processed
    )
else:
    np.save(
        PROCESSED_DIR / "X_train.npy",
        X_train_processed
    )

# Save testing features
if issparse(X_test_processed):
    save_npz(
        PROCESSED_DIR / "X_test.npz",
        X_test_processed
    )
else:
    np.save(
        PROCESSED_DIR / "X_test.npy",
        X_test_processed
    )

print("Processed datasets saved successfully!")

y_train.to_csv(
    PROCESSED_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    PROCESSED_DIR / "y_test.csv",
    index=False
)

print("\nPreprocessing completed!")

print("Training rows:", X_train_processed.shape[0])
print("Testing rows:", X_test_processed.shape[0])
print("Processed features:", X_train_processed.shape[1])
