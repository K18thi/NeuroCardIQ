import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
print("📦 Loading fused features...")
X = np.load("data/X_scaled.npy")
y = np.load("data/y.npy")
FEATURE_NAMES = np.load("data/feature_names.npy", allow_pickle=True).tolist()
RISK = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(f"  Classes : {np.unique(y)}")

# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION SETUP  (5-fold stratified)
# ─────────────────────────────────────────────────────────────
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

# ─────────────────────────────────────────────────────────────
# MODEL 1 — Random Forest
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*55)
print("🌲 Training Random Forest...")
print("═"*55)

rf = RandomForestClassifier(
    n_estimators   = 300,
    max_depth      = None,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    max_features   = 'sqrt',
    class_weight   = 'balanced',
    random_state   = 42,
    n_jobs         = -1
)

rf_cv = cross_validate(rf, X, y, cv=CV, scoring=SCORING, return_train_score=True)

print(f"\n  5-Fold CV Results:")
print(f"  {'Metric':<25} {'Mean':>8}  {'Std':>8}")
print(f"  {'─'*25} {'─'*8}  {'─'*8}")
for metric in SCORING:
    train_key = f'train_{metric}'
    test_key  = f'test_{metric}'
    print(f"  {'Train '+metric:<25} "
          f"{rf_cv[train_key].mean():>8.4f}  "
          f"{rf_cv[train_key].std():>8.4f}")
    print(f"  {'Val '+metric:<25} "
          f"{rf_cv[test_key].mean():>8.4f}  "
          f"{rf_cv[test_key].std():>8.4f}")
    print()

# Fit final RF on ALL data
rf.fit(X, y)
rf_preds = rf.predict(X)
print(f"  Final RF (full data) accuracy : {accuracy_score(y, rf_preds):.4f}")
print("\n  Classification Report (full data):")
print(classification_report(y, rf_preds,
      target_names=["Low Risk","Medium Risk","High Risk"]))

# ─────────────────────────────────────────────────────────────
# MODEL 2 — XGBoost
# ─────────────────────────────────────────────────────────────
print("═"*55)
print("⚡ Training XGBoost...")
print("═"*55)

xgb = XGBClassifier(
    n_estimators      = 300,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    use_label_encoder = False,
    eval_metric       = 'mlogloss',
    random_state      = 42,
    n_jobs            = -1
)

xgb_cv = cross_validate(xgb, X, y, cv=CV, scoring=SCORING, return_train_score=True)

print(f"\n  5-Fold CV Results:")
print(f"  {'Metric':<25} {'Mean':>8}  {'Std':>8}")
print(f"  {'─'*25} {'─'*8}  {'─'*8}")
for metric in SCORING:
    train_key = f'train_{metric}'
    test_key  = f'test_{metric}'
    print(f"  {'Train '+metric:<25} "
          f"{xgb_cv[train_key].mean():>8.4f}  "
          f"{xgb_cv[train_key].std():>8.4f}")
    print(f"  {'Val '+metric:<25} "
          f"{xgb_cv[test_key].mean():>8.4f}  "
          f"{xgb_cv[test_key].std():>8.4f}")
    print()

# Fit final XGB on ALL data
xgb.fit(X, y)
xgb_preds = xgb.predict(X)
print(f"  Final XGB (full data) accuracy : {accuracy_score(y, xgb_preds):.4f}")
print("\n  Classification Report (full data):")
print(classification_report(y, xgb_preds,
      target_names=["Low Risk","Medium Risk","High Risk"]))

# ─────────────────────────────────────────────────────────────
# COMPARE MODELS
# ─────────────────────────────────────────────────────────────
print("═"*55)
print("📊 Model Comparison (5-Fold CV Validation):")
print("═"*55)
rf_acc  = rf_cv['test_accuracy'].mean()
xgb_acc = xgb_cv['test_accuracy'].mean()
rf_f1   = rf_cv['test_f1_macro'].mean()
xgb_f1  = xgb_cv['test_f1_macro'].mean()

print(f"  {'Model':<15} {'CV Accuracy':>12}  {'CV F1-Macro':>12}")
print(f"  {'─'*15} {'─'*12}  {'─'*12}")
print(f"  {'Random Forest':<15} {rf_acc:>12.4f}  {rf_f1:>12.4f}")
print(f"  {'XGBoost':<15} {xgb_acc:>12.4f}  {xgb_f1:>12.4f}")

best_model      = rf   if rf_acc >= xgb_acc else xgb
best_model_name = "RandomForest" if rf_acc >= xgb_acc else "XGBoost"
print(f"\n  🏆 Best model : {best_model_name} "
      f"(CV acc = {max(rf_acc, xgb_acc):.4f})")

# ─────────────────────────────────────────────────────────────
# CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────
print("\n📋 Confusion Matrix — Random Forest (full data):")
print(confusion_matrix(y, rf_preds))
print("\n📋 Confusion Matrix — XGBoost (full data):")
print(confusion_matrix(y, xgb_preds))

# ─────────────────────────────────────────────────────────────
# TOP FEATURE IMPORTANCES
# ─────────────────────────────────────────────────────────────
print("\n🔍 Top 15 Most Important Features (Random Forest):")
rf_importances = rf.feature_importances_
top15_idx = np.argsort(rf_importances)[::-1][:15]
for rank, idx in enumerate(top15_idx, 1):
    print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<40s}  "
          f"{rf_importances[idx]:.4f}")

print("\n🔍 Top 15 Most Important Features (XGBoost):")
xgb_importances = xgb.feature_importances_
top15_idx_xgb = np.argsort(xgb_importances)[::-1][:15]
for rank, idx in enumerate(top15_idx_xgb, 1):
    print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<40s}  "
          f"{xgb_importances[idx]:.4f}")

# ─────────────────────────────────────────────────────────────
# SAVE BOTH MODELS + BEST MODEL FLAG
# ─────────────────────────────────────────────────────────────
joblib.dump(rf,   "models/random_forest.pkl")
joblib.dump(xgb,  "models/xgboost.pkl")
joblib.dump(best_model, "models/best_model.pkl")

# Save metadata for inference
meta = {
    "best_model_name" : best_model_name,
    "rf_cv_accuracy"  : round(rf_acc, 4),
    "xgb_cv_accuracy" : round(xgb_acc, 4),
    "rf_cv_f1"        : round(rf_f1, 4),
    "xgb_cv_f1"       : round(xgb_f1, 4),
    "n_features"      : X.shape[1],
    "classes"         : [0, 1, 2],
    "class_names"     : ["Low Risk", "Medium Risk", "High Risk"]
}
joblib.dump(meta, "models/model_meta.pkl")

print("\n✅ Step 7 Complete! Saved:")
print("   models/random_forest.pkl")
print("   models/xgboost.pkl")
print("   models/best_model.pkl")
print("   models/model_meta.pkl")
print("   models/scaler.pkl  (from Step 6)")