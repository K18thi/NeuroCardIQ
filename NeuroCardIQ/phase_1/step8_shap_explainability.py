import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
print("📦 Loading model and data...")
X             = np.load("data/X_scaled.npy")
y             = np.load("data/y.npy")
FEATURE_NAMES = np.load("data/feature_names.npy", allow_pickle=True).tolist()
xgb           = joblib.load("models/xgboost.pkl")
rf            = joblib.load("models/random_forest.pkl")
meta          = joblib.load("models/model_meta.pkl")

RISK  = ["Low Risk", "Medium Risk", "High Risk"]
N_EEG = 27
N_ECG = 28
N_INT = 18

print(f"  X shape       : {X.shape}")
print(f"  Best model    : {meta['best_model_name']}")

# ─────────────────────────────────────────────────────────────
# SHAP — XGBoost (TreeExplainer — fast & exact)
# ─────────────────────────────────────────────────────────────
print("\n⚙️  Computing SHAP values for XGBoost...")
explainer_xgb  = shap.TreeExplainer(xgb)
shap_values_xgb = explainer_xgb.shap_values(X)  # list of 3 arrays (one per class)

print(f"  SHAP output type : {type(shap_values_xgb)}")
if isinstance(shap_values_xgb, list):
    print(f"  Classes          : {len(shap_values_xgb)}")
    print(f"  Shape per class  : {shap_values_xgb[0].shape}")
else:
    print(f"  SHAP shape       : {shap_values_xgb.shape}")

# ─────────────────────────────────────────────────────────────
# SHAP — Random Forest
# ─────────────────────────────────────────────────────────────
print("\n⚙️  Computing SHAP values for Random Forest...")
explainer_rf   = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X)

# ─────────────────────────────────────────────────────────────
# HELPER — mean absolute SHAP per feature (across all classes)
# ─────────────────────────────────────────────────────────────
def mean_abs_shap(shap_vals):
    if isinstance(shap_vals, list):
        # average over classes
        return np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif shap_vals.ndim == 3:
        return np.abs(shap_vals).mean(axis=(0, 2))
    else:
        return np.abs(shap_vals).mean(axis=0)

shap_importance_xgb = mean_abs_shap(shap_values_xgb)
shap_importance_rf  = mean_abs_shap(shap_values_rf)

# ─────────────────────────────────────────────────────────────
# BLOCK-LEVEL CONTRIBUTION
# Shows how much Brain / Heart / Interaction each contribute
# ─────────────────────────────────────────────────────────────
def block_contributions(importance, n_eeg, n_ecg, n_int):
    eeg_imp  = importance[:n_eeg].sum()
    ecg_imp  = importance[n_eeg:n_eeg+n_ecg].sum()
    int_imp  = importance[n_eeg+n_ecg:].sum()
    total    = eeg_imp + ecg_imp + int_imp
    return {
        "Brain (EEG)"          : eeg_imp / total * 100,
        "Heart (ECG/HRV)"      : ecg_imp / total * 100,
        "Brain-Heart Interact" : int_imp / total * 100,
    }

print("\n" + "═"*55)
print("🧠💓 Block Contributions — XGBoost SHAP")
print("═"*55)
contrib_xgb = block_contributions(shap_importance_xgb, N_EEG, N_ECG, N_INT)
for block, pct in contrib_xgb.items():
    bar = "█" * int(pct / 2)
    print(f"  {block:<25} {pct:5.1f}%  {bar}")

print("\n" + "═"*55)
print("🧠💓 Block Contributions — Random Forest SHAP")
print("═"*55)
contrib_rf = block_contributions(shap_importance_rf, N_EEG, N_ECG, N_INT)
for block, pct in contrib_rf.items():
    bar = "█" * int(pct / 2)
    print(f"  {block:<25} {pct:5.1f}%  {bar}")

# ─────────────────────────────────────────────────────────────
# TOP 20 FEATURES BY SHAP — XGBoost
# ─────────────────────────────────────────────────────────────
print("\n🔍 Top 20 Features by SHAP Importance (XGBoost):")
top20 = np.argsort(shap_importance_xgb)[::-1][:20]
for rank, idx in enumerate(top20, 1):
    block = ("EEG" if idx < N_EEG
             else "ECG" if idx < N_EEG + N_ECG
             else "INT")
    print(f"  {rank:2d}. [{block}] {FEATURE_NAMES[idx]:<40s} "
          f"{shap_importance_xgb[idx]:.4f}")

# ─────────────────────────────────────────────────────────────
# PER-CLASS SHAP CONTRIBUTION
# ─────────────────────────────────────────────────────────────
print("\n📊 Per-Class Block Contributions (XGBoost SHAP):")
if isinstance(shap_values_xgb, list):
    shap_list = shap_values_xgb
elif shap_values_xgb.ndim == 3:
    shap_list = [shap_values_xgb[:, :, c]
                 for c in range(shap_values_xgb.shape[2])]
else:
    shap_list = [shap_values_xgb]

for cls_idx, cls_name in enumerate(RISK):
    if cls_idx >= len(shap_list):
        break
    cls_imp = np.abs(shap_list[cls_idx]).mean(axis=0)
    c = block_contributions(cls_imp, N_EEG, N_ECG, N_INT)
    print(f"\n  {cls_name}:")
    for block, pct in c.items():
        print(f"    {block:<25} {pct:5.1f}%")

# ─────────────────────────────────────────────────────────────
# SAVE SHAP VALUES & PLOTS
# ─────────────────────────────────────────────────────────────
print("\n💾 Saving SHAP summary plot...")

# Use class-0 SHAP for summary (or mean across classes)
if isinstance(shap_values_xgb, list):
    shap_for_plot = shap_values_xgb[0]
elif shap_values_xgb.ndim == 3:
    shap_for_plot = shap_values_xgb[:, :, 0]
else:
    shap_for_plot = shap_values_xgb

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_for_plot, X,
    feature_names=FEATURE_NAMES,
    max_display=20,
    show=False
)
plt.title("SHAP Summary — XGBoost (Low Risk class)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_summary_xgb.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: outputs/shap_summary_xgb.png")

# Bar plot — mean absolute SHAP
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_for_plot, X,
    feature_names=FEATURE_NAMES,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title("SHAP Feature Importance — XGBoost", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_bar_xgb.png", dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: outputs/shap_bar_xgb.png")

# Save raw SHAP values
np.save("outputs/shap_values_xgb.npy",
        np.array(shap_values_xgb
                 if not isinstance(shap_values_xgb, list)
                 else np.stack(shap_values_xgb, axis=-1)))
joblib.dump(explainer_xgb, "models/shap_explainer_xgb.pkl")
joblib.dump(explainer_rf,  "models/shap_explainer_rf.pkl")

print("\n✅ Step 8 Complete! Saved:")
print("   outputs/shap_summary_xgb.png")
print("   outputs/shap_bar_xgb.png")
print("   outputs/shap_values_xgb.npy")
print("   models/shap_explainer_xgb.pkl")
print("   models/shap_explainer_rf.pkl")