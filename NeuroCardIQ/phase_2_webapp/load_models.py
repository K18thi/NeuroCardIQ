"""
Run this ONCE after deployment to copy models
from phase_1 into the webapp models folder.
This is only needed for local setup.
For Render, models are committed directly.
"""
import os
import shutil

src_models = '../phase_1/models'
dst_models = 'models'
src_data   = '../phase_1/data'
dst_data   = 'data'

os.makedirs(dst_models, exist_ok=True)
os.makedirs(dst_data,   exist_ok=True)

model_files = [
    'xgboost.pkl', 'random_forest.pkl',
    'best_model.pkl', 'scaler.pkl',
    'shap_explainer_xgb.pkl', 'model_meta.pkl',
]
data_files = [
    'eeg_feature_names.npy', 'ecg_feature_names.npy',
    'interaction_names.npy', 'feature_names.npy',
]

for f in model_files:
    src = os.path.join(src_models, f)
    dst = os.path.join(dst_models, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'✅ Copied {f}')
    else:
        print(f'⚠️  Missing {f}')

for f in data_files:
    src = os.path.join(src_data, f)
    dst = os.path.join(dst_data, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'✅ Copied {f}')
    else:
        print(f'⚠️  Missing {f}')

print('\nDone! Models ready for deployment.')