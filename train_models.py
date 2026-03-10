"""
Training script for Exoplanet Verification System
===================================================
Replicates the EXACT pipeline from kepler_analysis.ipynb:
  - Same key_columns selection (koi_smass filtered out — not in CSV)
  - Same preprocessing (dedup → select → dropna target → median impute → 3*IQR cap → engineer)
  - Same classification models (8) with exact hyperparameters
  - Same regression models (8) with exact hyperparameters
  - Correct scaled/unscaled handling per model
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from lightgbm import LGBMClassifier, LGBMRegressor

np.random.seed(42)
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "..", "stellar analytics", "cumulative.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING  (exact kepler_analysis.ipynb pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("TRAINING: Exoplanet Verification Models")
print("=" * 80)

print("\n[1/5] Loading & cleaning data …")
df = pd.read_csv(DATA_PATH)
print(f"  Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

df_clean = df.copy()
df_clean = df_clean.drop_duplicates()
print(f"  After dedup: {len(df_clean):,} rows")

# Notebook key_columns — koi_smass is listed but NOT in CSV, so filtered out
key_columns = [
    "koi_disposition", "koi_score", "koi_period", "koi_teq", "koi_insol",
    "koi_depth", "koi_duration", "koi_prad", "koi_srad", "koi_smass",
    "koi_steff", "ra", "dec",
]
available_columns = [c for c in key_columns if c in df_clean.columns]
df_clean = df_clean[available_columns]
print(f"  Selected {len(available_columns)} columns (koi_smass absent — expected)")

# Drop rows with missing target
df_clean = df_clean.dropna(subset=["koi_disposition"])

# Median imputation for numerical columns
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
medians = {}
for col in numerical_cols:
    med = df_clean[col].median()
    medians[col] = float(med)
    df_clean[col] = df_clean[col].fillna(med)

# Outlier capping (3×IQR, skip ra/dec) — exact notebook logic
cap_cols = [c for c in numerical_cols if c not in ["ra", "dec"]]
iqr_bounds = {}
for col in cap_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = float(Q1 - 3 * IQR), float(Q3 + 3 * IQR)
    iqr_bounds[col] = {"lower": lo, "upper": hi}
    df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)

# Feature engineering — exact notebook logic
df_clean["planet_star_radius_ratio"] = df_clean["koi_prad"] / df_clean["koi_srad"]
df_clean["period_temp_product"] = df_clean["koi_period"] * df_clean["koi_teq"]
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

# Store medians for engineered features
medians["planet_star_radius_ratio"] = float(df_clean["planet_star_radius_ratio"].median())
medians["period_temp_product"] = float(df_clean["period_temp_product"].median())

print(f"  Cleaned dataset: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"  Disposition counts:\n{df_clean['koi_disposition'].value_counts().to_string()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION  (8 models — exact notebook)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Training classification models …")

# Use only CONFIRMED vs FALSE POSITIVE — exclude CANDIDATE to avoid
# treating unresolved candidates as false positives (which biases against
# small planets that are real but not yet confirmed).
df_model = df_clean[df_clean["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
df_model["target"] = (df_model["koi_disposition"] == "CONFIRMED").astype(int)

clf_feature_cols = [c for c in df_model.columns if c not in ["koi_disposition", "target"]]
X = df_model[clf_feature_cols].copy()
y = df_model["target"]

# Safety: handle NaN / Inf / constant columns
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median())
X = X.fillna(0)
constant = [c for c in X.columns if X[c].nunique() <= 1]
if constant:
    X = X.drop(columns=constant)
    clf_feature_cols = [c for c in clf_feature_cols if c not in constant]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf_scaler = StandardScaler()
X_train_sc = np.nan_to_num(clf_scaler.fit_transform(X_train), nan=0.0)
X_test_sc = np.nan_to_num(clf_scaler.transform(X_test), nan=0.0)

print(f"  Features ({len(clf_feature_cols)}): {clf_feature_cols}")
print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"  Target balance: {dict(y.value_counts())}")

clf_info = []

def _train_clf(name, model, scaled):
    X_tr = X_train_sc if scaled else X_train.values
    X_te = X_test_sc if scaled else X_test.values
    model.fit(X_tr, y_train)
    yp = model.predict(X_te)
    yprob = model.predict_proba(X_te)[:, 1]
    m = {
        "accuracy": round(accuracy_score(y_test, yp), 4),
        "precision": round(precision_score(y_test, yp), 4),
        "recall": round(recall_score(y_test, yp), 4),
        "f1": round(f1_score(y_test, yp), 4),
        "roc_auc": round(roc_auc_score(y_test, yprob), 4),
    }
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"clf_{safe}.joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))
    clf_info.append({"name": name, "file": fname, "scaled": scaled, "metrics": m})
    print(f"    {name:30s}  Acc={m['accuracy']}  F1={m['f1']}  AUC={m['roc_auc']}")

# ── Models with EXACT notebook params ──
# Scaled models: LR, KNN, MLP, SVM, Voting Ensemble
# Unscaled models: RF, GB, LightGBM
_train_clf("Logistic Regression",
           LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"), scaled=True)
_train_clf("Random Forest",
           RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"), scaled=False)
_train_clf("Gradient Boosting",
           GradientBoostingClassifier(n_estimators=100, random_state=42), scaled=False)
_train_clf("KNN",
           KNeighborsClassifier(n_neighbors=7, n_jobs=-1), scaled=True)
_train_clf("Neural Network (MLP)",
           MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                         random_state=42, early_stopping=True, validation_fraction=0.1),
           scaled=True)
_train_clf("SVM",
           SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"), scaled=True)
_train_clf("LightGBM",
           LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1,
                          is_unbalance=True), scaled=False)

# Voting Ensemble — fresh estimators, trained on SCALED data (exact notebook)
voting_clf = VotingClassifier(
    estimators=[
        ("lr",  LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")),
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")),
        ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                              random_state=42, early_stopping=True, validation_fraction=0.1)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")),
        ("lgb", LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, is_unbalance=True)),
    ],
    voting="soft",
    n_jobs=-1,
)
_train_clf("Voting Ensemble", voting_clf, scaled=True)

joblib.dump(clf_scaler, os.path.join(MODEL_DIR, "clf_scaler.joblib"))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. REGRESSION  (8 models — exact notebook, CONFIRMED only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Training regression models (CONFIRMED only) …")

df_reg = df_clean[df_clean["koi_disposition"] == "CONFIRMED"].copy()
# Exclude koi_prad (target) AND planet_star_radius_ratio (= koi_prad/koi_srad → data leakage)
reg_feature_cols = [c for c in df_reg.columns if c not in ["koi_disposition", "koi_prad", "planet_star_radius_ratio"]]

X_reg = df_reg[reg_feature_cols].copy()
y_reg = df_reg["koi_prad"].copy()

X_reg.replace([np.inf, -np.inf], np.nan, inplace=True)
X_reg = X_reg.fillna(X_reg.median())
X_reg = X_reg.fillna(0)

valid = y_reg.notna()
X_reg = X_reg.loc[valid].reset_index(drop=True)
y_reg = y_reg.loc[valid].reset_index(drop=True)

constant_reg = [c for c in X_reg.columns if X_reg[c].nunique() <= 1]
if constant_reg:
    X_reg = X_reg.drop(columns=constant_reg)
    reg_feature_cols = [c for c in reg_feature_cols if c not in constant_reg]

X_rtrain, X_rtest, y_rtrain, y_rtest = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
reg_scaler = StandardScaler()
X_rtrain_sc = np.nan_to_num(reg_scaler.fit_transform(X_rtrain), nan=0.0)
X_rtest_sc = np.nan_to_num(reg_scaler.transform(X_rtest), nan=0.0)

print(f"  Samples: {len(X_reg):,} confirmed exoplanets")
print(f"  Features ({len(reg_feature_cols)}): {reg_feature_cols}")
print(f"  Target koi_prad:  mean={y_reg.mean():.3f}  median={y_reg.median():.3f}  std={y_reg.std():.3f}")
print(f"  Train: {X_rtrain.shape[0]:,}  |  Test: {X_rtest.shape[0]:,}")

reg_info = []

def _train_reg(name, model, scaled):
    X_tr = X_rtrain_sc if scaled else X_rtrain.values
    X_te = X_rtest_sc if scaled else X_rtest.values
    model.fit(X_tr, y_rtrain)
    yp = model.predict(X_te)
    m = {
        "r2": round(r2_score(y_rtest, yp), 4),
        "mae": round(mean_absolute_error(y_rtest, yp), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_rtest, yp))), 4),
    }
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"reg_{safe}.joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))
    reg_info.append({"name": name, "file": fname, "scaled": scaled, "metrics": m})
    print(f"    {name:30s}  R²={m['r2']}  MAE={m['mae']}  RMSE={m['rmse']}")

# ── Regression Models (EXACT notebook params) ──
# Scaled: Linear, Ridge, Lasso, KNN, MLP
# Unscaled: RF, GB, LightGBM
_train_reg("Linear Regression", LinearRegression(), scaled=True)
_train_reg("Ridge Regression", Ridge(alpha=1.0, random_state=42), scaled=True)
_train_reg("Lasso Regression", Lasso(alpha=0.01, random_state=42, max_iter=5000), scaled=True)
_train_reg("KNN Regressor", KNeighborsRegressor(n_neighbors=7, n_jobs=-1), scaled=True)
_train_reg("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), scaled=False)
_train_reg("Gradient Boosting",
           GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42), scaled=False)
_train_reg("LightGBM", LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1), scaled=False)
_train_reg("Neural Network (MLP)",
           MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                        random_state=42, early_stopping=True, validation_fraction=0.1),
           scaled=True)

joblib.dump(reg_scaler, os.path.join(MODEL_DIR, "reg_scaler.joblib"))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SAVE METADATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Saving metadata …")

# Compute feature statistics from training data for UI context
feature_stats = {}
for col in clf_feature_cols:
    s = df_clean[col] if col in df_clean.columns else pd.Series()
    if len(s) > 0:
        feature_stats[col] = {
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
        }

metadata = {
    "clf_feature_cols": clf_feature_cols,
    "reg_feature_cols": reg_feature_cols,
    "clf_models": clf_info,
    "reg_models": reg_info,
    "medians": medians,
    "iqr_bounds": iqr_bounds,
    "feature_stats": feature_stats,
    "dataset_info": {
        "total_rows": len(df_clean),
        "confirmed": int((df_clean["koi_disposition"] == "CONFIRMED").sum()),
        "false_positive": int((df_clean["koi_disposition"] == "FALSE POSITIVE").sum()),
        "candidate": int((df_clean["koi_disposition"] == "CANDIDATE").sum()),
    },
    "regression_target_stats": {
        "count": int(len(y_reg)),
        "mean": round(float(y_reg.mean()), 4),
        "median": round(float(y_reg.median()), 4),
        "std": round(float(y_reg.std()), 4),
        "min": round(float(y_reg.min()), 4),
        "max": round(float(y_reg.max()), 4),
    },
}

with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Summary")
print("=" * 80)

best_clf = max(clf_info, key=lambda x: x["metrics"]["roc_auc"])
best_reg = max(reg_info, key=lambda x: x["metrics"]["r2"])
print(f"  Best classifier:  {best_clf['name']}  (AUC={best_clf['metrics']['roc_auc']})")
print(f"  Best regressor:   {best_reg['name']}  (R²={best_reg['metrics']['r2']})")
print(f"  Models saved to:  {MODEL_DIR}")
print(f"  Metadata:         {os.path.join(MODEL_DIR, 'metadata.json')}")
print("=" * 80)
print("TRAINING COMPLETE")
