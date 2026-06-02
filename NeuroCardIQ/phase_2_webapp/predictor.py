import numpy as np
import joblib
import os
import sys

# Add phase_1 models path
BASE_DIR = os.path.dirname(__file__)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

sys.path.insert(0, os.path.dirname(__file__))
from image_to_signal import (image_to_signal,
                              extract_eeg_features_from_signal,
                              extract_ecg_features_from_signal)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS (once at import time)
# ─────────────────────────────────────────────────────────────
print("📦 Loading models into predictor...")
scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
best_model    = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
rf_model      = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
xgb_model     = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
explainer     = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer_xgb.pkl'))
meta          = joblib.load(os.path.join(MODELS_DIR, 'model_meta.pkl'))

EEG_NAMES     = np.load(os.path.join(DATA_DIR, 'eeg_feature_names.npy'),
                         allow_pickle=True).tolist()
ECG_NAMES     = np.load(os.path.join(DATA_DIR, 'ecg_feature_names.npy'),
                         allow_pickle=True).tolist()
INTERACT_NAMES= np.load(os.path.join(DATA_DIR, 'interaction_names.npy'),
                         allow_pickle=True).tolist()
FEATURE_NAMES = EEG_NAMES + ECG_NAMES + INTERACT_NAMES

RISK_NAMES    = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS   = {0: "#2ecc71",  1: "#f39c12",     2: "#e74c3c"}
RISK_ICONS    = {0: "🟢",       1: "🟡",           2: "🔴"}

N_EEG = len(EEG_NAMES)   # 27
N_ECG = len(ECG_NAMES)   # 28
N_INT = len(INTERACT_NAMES)  # 18

print(f"  ✅ Models loaded | Best: {meta['best_model_name']} "
      f"| CV acc: {meta['xgb_cv_accuracy']}")


# ─────────────────────────────────────────────────────────────
# STEP A — Build feature vector from images
# ─────────────────────────────────────────────────────────────
def build_feature_vector(eeg_image_path, ecg_image_path):
    """
    Convert two waveform images → fused 73-dim feature vector.
    Returns (X_raw, eeg_features_dict, ecg_features_dict)
    """
    # 1. Image → signal
    eeg_signal = image_to_signal(eeg_image_path, target_length=500)
    ecg_signal = image_to_signal(ecg_image_path, target_length=500)

    # 2. Signal → features
    eeg_feat_dict = extract_eeg_features_from_signal(eeg_signal)
    ecg_feat_dict = extract_ecg_features_from_signal(ecg_signal)

    # 3. Align to exact training column order
    eeg_vec = np.array([eeg_feat_dict.get(n, 0.0) for n in EEG_NAMES],
                        dtype=np.float32)
    ecg_vec = np.array([ecg_feat_dict.get(n, 0.0) for n in ECG_NAMES],
                        dtype=np.float32)

    # 4. Brain-Heart interaction (mirrors step5)
    from scipy.stats import pearsonr
    from scipy.signal import correlate

    # Pearson
    eeg_trim = eeg_vec[:N_ECG] if N_EEG > N_ECG else eeg_vec
    ecg_trim = ecg_vec[:len(eeg_trim)]
    try:
        pearson_val, _ = pearsonr(eeg_trim, ecg_trim)
        pearson_val = 0.0 if np.isnan(pearson_val) else float(pearson_val)
    except Exception:
        pearson_val = 0.0

    # Cross-correlation
    eeg_n = eeg_trim - eeg_trim.mean()
    ecg_n = ecg_trim - ecg_trim.mean()
    xcorr = correlate(eeg_n, ecg_n, mode='full')
    xcorr_peak = float(np.max(np.abs(xcorr)))
    xcorr_lag  = float(np.argmax(np.abs(xcorr)) - (len(eeg_trim) - 1))

    # Band × HRV interaction
    bands = {
        'alpha': [i for i, n in enumerate(EEG_NAMES) if 'alpha' in n],
        'beta' : [i for i, n in enumerate(EEG_NAMES) if 'beta'  in n],
        'theta': [i for i, n in enumerate(EEG_NAMES) if 'theta' in n],
    }
    hrv_keys = ['HRV_LFHF', 'HRV_SD2', 'HRV_HF', 'HRV_ShanEn', 'HRV_ApEn']

    band_hrv = []
    for band_name, band_idx in bands.items():
        band_mean = eeg_vec[band_idx].mean()
        for hk in hrv_keys:
            if hk in ECG_NAMES:
                hi = ECG_NAMES.index(hk)
                band_hrv.append(float(band_mean * ecg_vec[hi]))
            else:
                band_hrv.append(0.0)

    interact_vec = np.array(
        [pearson_val, xcorr_peak, xcorr_lag] + band_hrv,
        dtype=np.float32
    )

    # 5. Fuse
    X_raw = np.hstack([eeg_vec, ecg_vec, interact_vec]).reshape(1, -1)
    return X_raw, eeg_feat_dict, ecg_feat_dict


# ─────────────────────────────────────────────────────────────
# STEP B — Predict risk
# ─────────────────────────────────────────────────────────────
def predict_risk(eeg_image_path, ecg_image_path):
    """
    Full pipeline: images → prediction + explanation.
    Returns a rich result dict.
    """
    # Build features
    X_raw, eeg_feats, ecg_feats = build_feature_vector(
        eeg_image_path, ecg_image_path)

    # Clip outliers
    means = X_raw.mean()
    X_clipped = np.clip(X_raw, -10, 10)

    # Scale
    X_scaled = scaler.transform(X_clipped)

    # Predict — both models
    xgb_pred  = int(xgb_model.predict(X_scaled)[0])
    rf_pred   = int(rf_model.predict(X_scaled)[0])
    best_pred = int(best_model.predict(X_scaled)[0])

    xgb_proba  = xgb_model.predict_proba(X_scaled)[0]
    rf_proba   = rf_model.predict_proba(X_scaled)[0]
    best_proba = best_model.predict_proba(X_scaled)[0]

    confidence = float(best_proba[best_pred]) * 100

    # SHAP explanation
    shap_vals = explainer.shap_values(X_scaled)   # (1, 73, 3)
    if shap_vals.ndim == 3:
        shap_for_class = shap_vals[0, :, best_pred]
    else:
        shap_for_class = shap_vals[0]

    # Block contributions
    eeg_contrib  = float(np.abs(shap_for_class[:N_EEG]).sum())
    ecg_contrib  = float(np.abs(shap_for_class[N_EEG:N_EEG+N_ECG]).sum())
    int_contrib  = float(np.abs(shap_for_class[N_EEG+N_ECG:]).sum())
    total_contrib = eeg_contrib + ecg_contrib + int_contrib + 1e-9

    block_pct = {
        "brain"       : round(eeg_contrib  / total_contrib * 100, 1),
        "heart"       : round(ecg_contrib  / total_contrib * 100, 1),
        "interaction" : round(int_contrib  / total_contrib * 100, 1),
    }

    # Top 10 features by SHAP
    top10_idx = np.argsort(np.abs(shap_for_class))[::-1][:10]
    top_features = [
        {
            "name"      : FEATURE_NAMES[i],
            "shap"      : round(float(shap_for_class[i]), 4),
            "abs_shap"  : round(float(abs(shap_for_class[i])), 4),
            "block"     : ("EEG" if i < N_EEG
                           else "ECG" if i < N_EEG + N_ECG
                           else "INT"),
            "direction" : "↑ increases risk" if shap_for_class[i] > 0
                          else "↓ decreases risk"
        }
        for i in top10_idx
    ]

    # Agreement between models
    models_agree = (xgb_pred == rf_pred)

    result = {
        # Core prediction
        "risk_level"      : best_pred,
        "risk_name"       : RISK_NAMES[best_pred],
        "risk_color"      : RISK_COLORS[best_pred],
        "risk_icon"       : RISK_ICONS[best_pred],
        "confidence"      : round(confidence, 1),

        # Probabilities
        "prob_low"        : round(float(best_proba[0]) * 100, 1),
        "prob_medium"     : round(float(best_proba[1]) * 100, 1),
        "prob_high"       : round(float(best_proba[2]) * 100, 1),

        # Both models
        "xgb_prediction"  : RISK_NAMES[xgb_pred],
        "rf_prediction"   : RISK_NAMES[rf_pred],
        "models_agree"    : models_agree,

        # SHAP explainability
        "block_pct"       : block_pct,
        "top_features"    : top_features,

        # Raw features (for display)
        "eeg_summary"     : {
            "alpha_mean": round(float(np.mean([v for k,v in eeg_feats.items()
                                               if 'alpha' in k])), 4),
            "beta_mean" : round(float(np.mean([v for k,v in eeg_feats.items()
                                               if 'beta'  in k])), 4),
            "theta_mean": round(float(np.mean([v for k,v in eeg_feats.items()
                                               if 'theta' in k])), 4),
        },
        "ecg_summary"     : {
            "r_maximum" : round(float(ecg_feats.get('r_maximum', 0)), 4),
            "HRV_LFHF"  : round(float(ecg_feats.get('HRV_LFHF',  0)), 4),
            "HRV_SD2"   : round(float(ecg_feats.get('HRV_SD2',   0)), 4),
        },

        # Model metadata
        "model_name"      : meta['best_model_name'],
        "model_accuracy"  : meta['xgb_cv_accuracy'],
    }

    return result


# ─────────────────────────────────────────────────────────────
# TEST predictor
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_EEG = "test_images/eeg_sample.png"
    TEST_ECG = "test_images/ecg_sample.png"

    print("\n" + "═"*55)
    print("🔮 Running full prediction pipeline...")
    print("═"*55)

    result = predict_risk(TEST_EEG, TEST_ECG)

    print(f"\n  {result['risk_icon']}  Risk Level   : {result['risk_name']}")
    print(f"  📊 Confidence  : {result['confidence']}%")
    print(f"  🤝 Models agree: {result['models_agree']} "
          f"(XGB={result['xgb_prediction']}, RF={result['rf_prediction']})")

    print(f"\n  Probabilities:")
    print(f"    Low    : {result['prob_low']}%")
    print(f"    Medium : {result['prob_medium']}%")
    print(f"    High   : {result['prob_high']}%")

    print(f"\n  🧠💓 Block Contributions:")
    for block, pct in result['block_pct'].items():
        bar = "█" * int(pct / 4)
        print(f"    {block:<12} : {pct:5.1f}%  {bar}")

    print(f"\n  🔍 Top 5 Features (SHAP):")
    for i, f in enumerate(result['top_features'][:5], 1):
        print(f"    {i}. [{f['block']}] {f['name']:<35} "
              f"SHAP={f['shap']:+.4f}  {f['direction']}")

    print(f"\n  🧠 EEG Summary : {result['eeg_summary']}")
    print(f"  💓 ECG Summary : {result['ecg_summary']}")

    print(f"\n✅ Step 10 Complete! Predictor is working.")
