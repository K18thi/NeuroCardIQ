import numpy as np
#import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# LOAD saved arrays from Step 1
# ─────────────────────────────────────────────────────────────
print("📦 Loading data...")
X_eeg        = np.load("data/X_eeg.npy")
X_ecg        = np.load("data/X_ecg.npy")
y            = np.load("data/y.npy")
EEG_FEATURES = np.load("data/eeg_feature_names.npy", allow_pickle=True).tolist()
ECG_FEATURES = np.load("data/ecg_feature_names.npy", allow_pickle=True).tolist()

print(f"  X_eeg : {X_eeg.shape}")
print(f"  X_ecg : {X_ecg.shape}")

N_SAMPLES = X_eeg.shape[0]   # 414
N_EEG     = X_eeg.shape[1]   # 27
N_ECG     = X_ecg.shape[1]   # 28

# ─────────────────────────────────────────────────────────────
# INTERACTION FEATURE 1 — Pearson Correlation (per sample)
# For each sample: correlate the 27-dim EEG vector
# with the 28-dim ECG vector (use the min length = 27)
# ─────────────────────────────────────────────────────────────
print("\n🔗 Computing Pearson correlations (EEG ↔ ECG per sample)...")

pearson_features = []
for i in range(N_SAMPLES):
    eeg_vec = X_eeg[i]          # shape (27,)
    ecg_vec = X_ecg[i, :N_EEG]  # shape (27,) — match length

    corr, pval = pearsonr(eeg_vec, ecg_vec)
    # Replace NaN (constant vectors) with 0
    corr = 0.0 if np.isnan(corr) else corr
    pearson_features.append(corr)

pearson_features = np.array(pearson_features).reshape(-1, 1)
print(f"  ✅ Pearson feature shape : {pearson_features.shape}")
print(f"     Mean corr = {pearson_features.mean():.4f} | "
      f"Std = {pearson_features.std():.4f} | "
      f"Range = [{pearson_features.min():.4f}, {pearson_features.max():.4f}]")

# ─────────────────────────────────────────────────────────────
# INTERACTION FEATURE 2 — Cross-Correlation Peak & Lag
# Captures the TIME-SHIFT relationship between EEG and ECG
# ─────────────────────────────────────────────────────────────
print("\n⏱️  Computing cross-correlation (peak & lag)...")

xcorr_peak_features = []
xcorr_lag_features  = []

for i in range(N_SAMPLES):
    eeg_vec = X_eeg[i]
    ecg_vec = X_ecg[i, :N_EEG]

    # Normalize to zero mean before cross-correlation
    eeg_norm = eeg_vec - eeg_vec.mean()
    ecg_norm = ecg_vec - ecg_vec.mean()

    # Full cross-correlation
    xcorr = correlate(eeg_norm, ecg_norm, mode='full')

    # Peak value (max absolute correlation)
    peak_val = np.max(np.abs(xcorr))

    # Lag at peak
    lag = np.argmax(np.abs(xcorr)) - (N_EEG - 1)

    xcorr_peak_features.append(peak_val)
    xcorr_lag_features.append(lag)

xcorr_peak_features = np.array(xcorr_peak_features).reshape(-1, 1)
xcorr_lag_features  = np.array(xcorr_lag_features, dtype=np.float32).reshape(-1, 1)

print(f"  ✅ XCorr Peak shape : {xcorr_peak_features.shape}")
print(f"     Mean peak = {xcorr_peak_features.mean():.4f}")
print(f"  ✅ XCorr Lag  shape : {xcorr_lag_features.shape}")
print(f"     Mean lag  = {xcorr_lag_features.mean():.2f}")

# ─────────────────────────────────────────────────────────────
# INTERACTION FEATURE 3 — Band-specific EEG × HRV products
# Multiply each EEG band mean × each HRV feature
# Gives the model direct brain-heart coupling signals
# ─────────────────────────────────────────────────────────────
print("\n🧠💓 Computing band-level EEG × HRV interaction features...")

# Group EEG columns by band
bands = {
    'alpha': [i for i, n in enumerate(EEG_FEATURES) if 'alpha' in n],
    'beta' : [i for i, n in enumerate(EEG_FEATURES) if 'beta'  in n],
    'theta': [i for i, n in enumerate(EEG_FEATURES) if 'theta' in n],
}

# Key HRV features to interact with
hrv_keys   = ['HRV_LFHF', 'HRV_SD2', 'HRV_HF', 'HRV_ShanEn', 'HRV_ApEn']
hrv_indices = [ECG_FEATURES.index(k) for k in hrv_keys if k in ECG_FEATURES]
hrv_names   = [ECG_FEATURES[i] for i in hrv_indices]

print(f"  EEG bands   : {[(b, len(v)) for b,v in bands.items()]}")
print(f"  HRV targets : {hrv_names}")

band_hrv_features = []
band_hrv_names    = []

for band_name, band_indices in bands.items():
    band_mean = X_eeg[:, band_indices].mean(axis=1)  # (414,)
    for hi, hname in zip(hrv_indices, hrv_names):
        hrv_col   = X_ecg[:, hi]                     # (414,)
        product   = band_mean * hrv_col               # element-wise
        band_hrv_features.append(product)
        band_hrv_names.append(f"interact_{band_name}_{hname}")

band_hrv_features = np.column_stack(band_hrv_features)  # (414, 15)
print(f"  ✅ Band×HRV feature shape : {band_hrv_features.shape}")
print(f"     Feature names: {band_hrv_names}")

# ─────────────────────────────────────────────────────────────
# COMBINE ALL INTERACTION FEATURES
# ─────────────────────────────────────────────────────────────
X_interaction = np.hstack([
    pearson_features,       # (414, 1)
    xcorr_peak_features,    # (414, 1)
    xcorr_lag_features,     # (414, 1)
    band_hrv_features       # (414, 15)
])

INTERACTION_NAMES = (
    ['pearson_eeg_ecg', 'xcorr_peak', 'xcorr_lag']
    + band_hrv_names
)

print(f"\n✅ Final Interaction Feature Matrix : {X_interaction.shape}")
print(f"   Total interaction features       : {len(INTERACTION_NAMES)}")
for n in INTERACTION_NAMES:
    print(f"   • {n}")

# ─────────────────────────────────────────────────────────────
# QUICK SANITY — correlation with risk label
# ─────────────────────────────────────────────────────────────
print("\n📊 Interaction features vs Risk Label (Pearson r):")
for i, name in enumerate(INTERACTION_NAMES):
    r, p = pearsonr(X_interaction[:, i], y)
    sig = "⭐" if p < 0.05 else "  "
    print(f"  {sig} {name:45s}  r={r:+.3f}  p={p:.4f}")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
np.save("data/X_interaction.npy",      X_interaction)
np.save("data/interaction_names.npy",  np.array(INTERACTION_NAMES))

print("\n✅ Step 5 Complete! Saved:")
print("   data/X_interaction.npy")
print("   data/interaction_names.npy")