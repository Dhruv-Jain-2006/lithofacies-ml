import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def augment_rare_facies(df_train, ratio=0.10, random_state=42):
    """
    Extracts contiguous sequence windows of rare classes (Anhydrite: 8, Tuff: 9, Coal: 10, Basement: 11)
    from the training split, applies realistic, physically-sound geological perturbations,
    and appends them strictly to the training set.
    
    Perturbations applied:
    - Gamma Ray (GR): +/- 5% scaling (multiplicative noise)
    - Density (RHOB): +/- 0.05 g/cm3 (additive noise)
    - Neutron Porosity (NPHI): +/- 0.03 (additive noise)
    - Travel Time (DTC): +/- 3.0 us/ft (additive noise)
    - Resistivities (RDEP, RMED, RSHA): log-multiplicative +/- 10%
    
    Petrophysical engineered features are mathematically recomputed for consistency.
    Rolling statistics and wavelets are perturbed with mild noise to maintain correlation structure.
    """
    if ratio <= 0.0:
        return df_train.copy()
        
    np.random.seed(random_state)
    df_out = df_train.copy()
    
    # Rare lithologies to target
    rare_classes = [8, 9, 10, 11]
    
    # Identify rows belonging to rare classes
    rare_df = df_train[df_train['LITHOLOGY'].isin(rare_classes)].reset_index(drop=True)
    
    if len(rare_df) == 0:
        logger.warning("No rare classes found in the training split for augmentation!")
        return df_out
        
    num_to_augment = int(len(df_train) * ratio)
    logger.info(f"Targeting augmentation of {num_to_augment} rare-facies samples (ratio: {ratio:.1%})...")
    
    # Bootstrap sample from the rare facies rows to generate the desired number of samples
    sampled_indices = np.random.choice(len(rare_df), size=num_to_augment, replace=True)
    augmented_df = rare_df.iloc[sampled_indices].copy()
    
    # Apply physically-sound geological perturbations
    n_samples = len(augmented_df)
    
    # 1. GR perturbation: +/- 5% scaling (Std Dev ~ 0.02)
    gr_perturb = 1.0 + np.random.normal(0, 0.02, size=n_samples)
    augmented_df['GR'] = augmented_df['GR'] * gr_perturb
    
    # 2. RHOB perturbation: +/- 0.05 g/cm3 (Std Dev ~ 0.015)
    rhob_perturb = np.random.normal(0, 0.015, size=n_samples)
    augmented_df['RHOB'] = augmented_df['RHOB'] + rhob_perturb
    
    # 3. NPHI perturbation: +/- 0.03 (Std Dev ~ 0.01)
    nphi_perturb = np.random.normal(0, 0.01, size=n_samples)
    augmented_df['NPHI'] = (augmented_df['NPHI'] + nphi_perturb).clip(0.0, 1.0)
    
    # 4. DTC perturbation: +/- 3.0 us/ft (Std Dev ~ 1.0)
    dtc_perturb = np.random.normal(0, 1.0, size=n_samples)
    augmented_df['DTC'] = augmented_df['DTC'] + dtc_perturb
    
    # 5. PEF perturbation: +/- 0.1 (Std Dev ~ 0.04)
    pef_perturb = np.random.normal(0, 0.04, size=n_samples)
    augmented_df['PEF'] = (augmented_df['PEF'] + pef_perturb).clip(0.1, 10.0)
    
    # 6. SP perturbation: +/- 2.0 mV
    sp_perturb = np.random.normal(0, 1.0, size=n_samples)
    augmented_df['SP'] = augmented_df['SP'] + sp_perturb
    
    # 7. Resistivity log-multiplicative perturbation +/- 10% (Std Dev ~ 0.03)
    for col in ['RDEP', 'RMED', 'RSHA']:
        if col in augmented_df.columns:
            r_perturb = np.exp(np.random.normal(0, 0.03, size=n_samples))
            augmented_df[col] = (augmented_df[col] * r_perturb).clip(lower=1e-5)
            
    # 8. Recompute core petrophysical engineered features for physical consistency
    if 'RHOB_minus_NPHI' in augmented_df.columns:
        augmented_df['RHOB_minus_NPHI'] = augmented_df['RHOB'] - augmented_df['NPHI']
    if 'RDEP_RMED_ratio' in augmented_df.columns:
        augmented_df['RDEP_RMED_ratio'] = augmented_df['RDEP'] / (augmented_df['RMED'] + 1e-5)
    if 'RMED_RSHA_ratio' in augmented_df.columns:
        augmented_df['RMED_RSHA_ratio'] = augmented_df['RMED'] / (augmented_df['RSHA'] + 1e-5)
    if 'acoustic_impedance' in augmented_df.columns:
        augmented_df['acoustic_impedance'] = augmented_df['RHOB'] * (304800 / (augmented_df['DTC'] + 1e-5))
        
    for col in ['RDEP', 'RMED', 'RSHA']:
        if f'log_{col}' in augmented_df.columns:
            augmented_df[f'log_{col}'] = np.log10(augmented_df[col].clip(lower=1e-5) + 1e-5)
            
    # 9. Add mild correlated noise to rolling features and wavelet features to preserve context structure
    # This prevents the classifier from over-relying on unmodified engineered bands
    for col in augmented_df.columns:
        if col in ['LITHOLOGY', 'WELL', 'WELL_ID', 'DEPTH_MD', 'IS_SYNTHETIC', 'RELATIVE_DEPTH', 
                   'GR', 'RHOB', 'NPHI', 'DTC', 'PEF', 'SP', 'RDEP', 'RMED', 'RSHA',
                   'RHOB_minus_NPHI', 'RDEP_RMED_ratio', 'RMED_RSHA_ratio', 'acoustic_impedance',
                   'log_RDEP', 'log_RMED', 'log_RSHA']:
            continue
        if pd.api.types.is_numeric_dtype(augmented_df[col]):
            # Add small noise (Std Dev ~ 1% of the value)
            col_std = df_train[col].std()
            if col_std > 0:
                noise = np.random.normal(0, 0.01 * col_std, size=n_samples)
                augmented_df[col] = augmented_df[col] + noise
                
    # Mark augmented rows as synthetic so they are identifiable
    augmented_df['IS_SYNTHETIC'] = True
    
    # Append the augmented rare classes strictly to the training set
    df_augmented = pd.concat([df_out, augmented_df], ignore_index=True)
    logger.info(f"Rare-facies augmentation completed. Added {len(augmented_df)} rows. New training set size: {len(df_augmented)}")
    
    return df_augmented
