# Advance Level Delhi Metro ML Pipeline
# -------------------------------------
# Steps:
# 1. Import libraries
# 2. Load CSV
# 3. EDA (shape, head, info, describe, dtypes, missing values, outliers)
# 4. Feature engineering
# 5. Train/test split
# 6. RandomForest model training
# 7. Evaluation (accuracy, confusion matrix, classification report)
# 8. Visualizations
# 9. Save model

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import joblib

# -------------------- Config --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "delhi_metro.csv")   # keep CSV in same folder
MODEL_PATH = os.path.join(BASE_DIR, "metro_interchange_rf.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------- Load dataset --------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("\n=== EDA: Basic Info ===")
print("Shape:", df.shape)
print("\nHead:\n", df.head(10))
print("\nDtypes:\n", df.dtypes)   # ✅ FIXED: no parentheses

# -------------------- EDA: describe + missing + outlier hint --------------------
print("\nDescribe (numeric):\n", df.select_dtypes(include=[np.number]).describe())

missing = df.isna().sum()
print("\nMissing values per column:\n", missing[missing > 0] if missing.sum() > 0 else "No missing values detected.")

if "fare" in df.columns:
    q1 = df["fare"].quantile(0.25)
    q3 = df["fare"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    suspected = df[(df["fare"] > upper) | (df["fare"] < lower)]
    print(f"\nOutlier hint (fare): IQR [{lower:.2f}, {upper:.2f}], suspected rows: {len(suspected)}")

# -------------------- Feature engineering --------------------
# Target: interchange trip
for col in ["origin_is_interchange", "destination_is_interchange"]:
    if col not in df.columns:
        df[col] = 0

df["is_interchange_trip"] = ((df["origin_is_interchange"].fillna(0) > 0) |
                             (df["destination_is_interchange"].fillna(0) > 0)).astype(int)

# Parse dates
df["trip_date_dt"] = pd.to_datetime(df.get("trip_date", pd.NaT), errors="coerce")
df["tap_in_dt"] = pd.to_datetime(df.get("tap_in_time", pd.NaT), errors="coerce")
df["tap_out_dt"] = pd.to_datetime(df.get("tap_out_time", pd.NaT), errors="coerce")

# Extract features
df["trip_year"] = df["trip_date_dt"].dt.year
df["trip_month"] = df["trip_date_dt"].dt.month
df["trip_day"] = df["trip_date_dt"].dt.day
df["trip_dayofweek"] = df["trip_date_dt"].dt.dayofweek
df["hour_in"] = df["tap_in_dt"].dt.hour
df["hour_out"] = df["tap_out_dt"].dt.hour
df["duration_min"] = (df["tap_out_dt"] - df["tap_in_dt"]).dt.total_seconds() / 60.0
df["is_peak"] = df["hour_in"].between(8, 10) | df["hour_in"].between(17, 20)
df["is_peak"] = df["is_peak"].astype(int)

# -------------------- Features and target --------------------
feature_cols_cat = ["origin_station", "destination_station", "origin_line"]
feature_cols_num = ["fare", "trip_year", "trip_month", "trip_day", "trip_dayofweek",
                    "hour_in", "hour_out", "duration_min", "is_peak"]

X = df[feature_cols_cat + feature_cols_num].copy()
y = df["is_interchange_trip"].copy()

# -------------------- Train/test split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -------------------- Preprocessing + model --------------------
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, feature_cols_cat),
        ("num", numeric_transformer, feature_cols_num)
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", model)
])

# -------------------- Train --------------------
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# -------------------- Evaluate --------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# -------------------- Visualizations --------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature importance
perm_result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)

feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
importances = perm_result.importances_mean
order = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[order], y=np.array(feature_names)[order], color="steelblue")
plt.title("Permutation Feature Importance (Top 20)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -------------------- Save model --------------------
joblib.dump(pipeline, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
