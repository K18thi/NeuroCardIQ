import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD all feature blocks
# ─────────────────────────────────────────────────────────────
print("📦 Loading feature blocks...")
X_eeg             = np.load("data/X_eeg.npy")
X_ecg             = np.load("data/X_ecg.npy")
X_interaction     = np.load("data/X_interaction.npy")
y                 = np.load("data/y.npy")
EEG_NAMES         = np.load("data/eeg_feature_names.npy",  allow_pickle=True).tolist()
ECG_NAMES         = np.load("data/ecg_feature_names.npy",  allow_pickle=True).tolist()
INTERACT_NAMES    = np.load("data/interaction_names.npy",  allow_pickle=True).tolist()

print(f"  EEG block        : {X_eeg.shape}")
print(f"  ECG block        : {X_ecg.shape}")
print(f"  Interaction block: {X_interaction.shape}")

# ─────────────────────────────────────────────────────────────
# FUSE — simple horizontal stack
# ─────────────────────────────────────────────────────────────
X_fused      = np.hstack([X_eeg, X_ecg, X_interaction])   # (414, 73)
FEATURE_NAMES = EEG_NAMES + ECG_NAMES + INTERACT_NAMES

print(f"\n🔀 Fused feature matrix : {X_fused.shape}")
print(f"   = {len(EEG_NAMES)} EEG  +  "
      f"{len(ECG_NAMES)} ECG  +  "
      f"{len(INTERACT_NAMES)} Interaction  "
      f"= {X_fused.shape[1]} total features")

# ─────────────────────────────────────────────────────────────
# HANDLE OUTLIERS — clip to ±5 std per feature
# ─────────────────────────────────────────────────────────────
print("\n✂️  Clipping outliers (±5σ)...")
means = X_fused.mean(axis=0)
stds  = X_fused.std(axis=0)
stds[stds == 0] = 1   # avoid div-by-zero
X_fused = np.clip(X_fused,
                  means - 5 * stds,
                  means + 5 * stds)
print("   Done.")

# ─────────────────────────────────────────────────────────────
# SCALE — StandardScaler on fused matrix
# ─────────────────────────────────────────────────────────────
print("\n📐 Scaling fused features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fused)

print(f"   Post-scale mean  : {X_scaled.mean():.6f}  (should be ≈ 0)")
print(f"   Post-scale std   : {X_scaled.std():.6f}   (should be ≈ 1)")
print(f"   Any NaN?         : {np.isnan(X_scaled).any()}")
print(f"   Any Inf?         : {np.isinf(X_scaled).any()}")

# ─────────────────────────────────────────────────────────────
# LABEL SUMMARY
# ─────────────────────────────────────────────────────────────
RISK = {0: "Low", 1: "Medium", 2: "High"}
print("\n🏷️  Label distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"   {RISK[u]:6s} Risk : {c} samples ({100*c/len(y):.1f}%)")

# ─────────────────────────────────────────────────────────────
# FEATURE BLOCK SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n📋 Feature Block Summary:")
print(f"   {'Block':<20} {'Features':>10}  {'Columns'}")
print(f"   {'─'*20} {'─'*10}  {'─'*30}")
print(f"   {'EEG (PSD bands)':<20} {len(EEG_NAMES):>10}  col 0–{len(EEG_NAMES)-1}")
print(f"   {'ECG (HRV+wave)':<20} {len(ECG_NAMES):>10}  col {len(EEG_NAMES)}–{len(EEG_NAMES)+len(ECG_NAMES)-1}")
print(f"   {'Brain-Heart':<20} {len(INTERACT_NAMES):>10}  col {len(EEG_NAMES)+len(ECG_NAMES)}–{X_fused.shape[1]-1}")
print(f"   {'─'*20} {'─'*10}")
print(f"   {'TOTAL':<20} {X_fused.shape[1]:>10}")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
np.save("data/X_scaled.npy",    X_scaled)
np.save("data/y.npy",           y)
np.save("data/feature_names.npy", np.array(FEATURE_NAMES))
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Step 6 Complete! Saved:")
print("   data/X_scaled.npy       ← final model input")
print("   data/feature_names.npy  ← all 73 feature names")
print("   models/scaler.pkl       ← scaler for inference")
