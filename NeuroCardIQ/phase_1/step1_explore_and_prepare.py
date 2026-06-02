import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# CONFIG — update paths to wherever you placed the CSVs
# ─────────────────────────────────────────────────────────────
EEG_PATH = "data/eeg.csv"
ECG_PATH = "data/ecg.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
print("📦 Loading datasets...")
eeg = pd.read_csv(EEG_PATH)
ecg = pd.read_csv(ECG_PATH)
print(f"  EEG shape: {eeg.shape}")
print(f"  ECG shape: {ecg.shape}")

META_COLS = ['Unnamed: 0', 'video_name', 'video', 'target']

EEG_FEATURES = [c for c in eeg.columns if c not in META_COLS]
ECG_FEATURES = [c for c in ecg.columns if c not in META_COLS]

print(f"\n📡 EEG features : {len(EEG_FEATURES)}")
print(f"💓 ECG features : {len(ECG_FEATURES)}")

def remap_risk(label):
    if label <= 2:
        return 0   # Low Risk
    elif label <= 5:
        return 1   # Medium Risk
    else:
        return 2   # High Risk

RISK_NAMES = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

eeg['risk'] = eeg['target'].apply(remap_risk)
ecg['risk'] = ecg['target'].apply(remap_risk)

print("\n📊 Risk Label Distribution (EEG):")
dist = eeg['risk'].value_counts().sort_index()
for k, v in dist.items():
    print(f"   {RISK_NAMES[k]:15s}: {v} samples ({100*v/len(eeg):.1f}%)")

# ─────────────────────────────────────────────────────────────
# VERIFY ALIGNMENT (both files must match row-for-row)
# ─────────────────────────────────────────────────────────────
assert (eeg['video'].values == ecg['video'].values).all(), \
    "❌ EEG and ECG rows are NOT aligned!"
assert (eeg['risk'].values == ecg['risk'].values).all(), \
    "❌ Risk labels differ between EEG and ECG!"
print("\n✅ EEG and ECG are perfectly aligned row-by-row")

# ─────────────────────────────────────────────────────────────
# EXTRACT CLEAN ARRAYS
# ─────────────────────────────────────────────────────────────
X_eeg  = eeg[EEG_FEATURES].values.astype(np.float32)  # (414, 27)
X_ecg  = ecg[ECG_FEATURES].values.astype(np.float32)  # (414, 28)
y      = eeg['risk'].values                             # (414,)

print(f"\n🔢 X_eeg  : {X_eeg.shape}")
print(f"🔢 X_ecg  : {X_ecg.shape}")
print(f"🏷️  Labels : {y.shape}  | Classes: {np.unique(y)}")

# ─────────────────────────────────────────────────────────────
# QUICK STATS — sanity check
# ─────────────────────────────────────────────────────────────
print("\n📈 EEG feature stats (first 5 cols):")
print(pd.DataFrame(X_eeg[:, :5],
      columns=EEG_FEATURES[:5]).describe().round(3))

print("\n💓 ECG feature stats (first 5 cols):")
print(pd.DataFrame(X_ecg[:, :5],
      columns=ECG_FEATURES[:5]).describe().round(3))

# Save feature names for later steps
np.save("data/eeg_feature_names.npy", np.array(EEG_FEATURES))
np.save("data/ecg_feature_names.npy", np.array(ECG_FEATURES))
np.save("data/X_eeg.npy",  X_eeg)
np.save("data/X_ecg.npy",  X_ecg)
np.save("data/y.npy",      y)

print("\n✅ Step 1 Complete! Saved:")
print("   data/X_eeg.npy, data/X_ecg.npy, data/y.npy")
print("   data/eeg_feature_names.npy, data/ecg_feature_names.npy")


