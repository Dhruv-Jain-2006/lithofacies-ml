import pickle
import os

original_cols_with_depth = ['RELATIVE_DEPTH', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']
original_cols_no_depth = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']

# Engineered features
wavelet_logs = ['GR', 'NPHI', 'SP', 'RDEP', 'RHOB', 'DTC', 'PEF']
petro_cols = ['RHOB_minus_NPHI', 'RDEP_RMED_ratio', 'RMED_RSHA_ratio', 'acoustic_impedance', 'log_RDEP', 'log_RMED', 'log_RSHA']
roll_cols = [
    'GR_roll_mean', 'RHOB_roll_mean', 'NPHI_roll_mean', 'DTC_roll_mean',
    'GR_roll_std', 'RHOB_roll_std', 'NPHI_roll_std', 'DTC_roll_std',
    'GR_roll_var', 'RHOB_roll_var',
    'GR_gradient', 'RHOB_gradient', 'DTC_gradient'
]
cwt_band_cols = []
for log in wavelet_logs:
    cwt_band_cols.extend([f"{log}_CWT_low", f"{log}_CWT_mid", f"{log}_CWT_high"])
engineered_cols = petro_cols + roll_cols + cwt_band_cols

# Load pruned_cols_with_depth from features_wavelet.pkl if exists
with open("data/models/features_wavelet.pkl", "rb") as f:
    pruned_cols_with_depth = pickle.load(f)

pruned_cols_no_depth = [c for c in pruned_cols_with_depth if c != 'RELATIVE_DEPTH']

# Save all of them
os.makedirs("data/models", exist_ok=True)
with open("data/models/features_original.pkl", "wb") as f:
    pickle.dump(original_cols_with_depth, f)
with open("data/models/features_original_nodep.pkl", "wb") as f:
    pickle.dump(original_cols_no_depth, f)
with open("data/models/features_engineered.pkl", "wb") as f:
    pickle.dump(engineered_cols, f)
with open("data/models/features_wavelet_nodep.pkl", "wb") as f:
    pickle.dump(pruned_cols_no_depth, f)

print("Successfully generated features pickle files!")
