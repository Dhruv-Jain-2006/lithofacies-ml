import os
import glob
import logging
import numpy as np
import pandas as pd
import lasio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class mappings as specified in the Implementation Plan
LITHOLOGY_MAP = {
    30000: 0,   # Sandstone
    65030: 1,   # Sandstone/Shale
    65000: 2,   # Shale
    80000: 3,   # Marl
    74000: 4,   # Dolomite
    70000: 5,   # Limestone
    70032: 6,   # Chalk
    88000: 7,   # Halite
    86000: 8,   # Anhydrite
    99000: 9,   # Tuff
    90000: 10,  # Coal
    93000: 11   # Basement
}

REVERSE_LITHOLOGY_MAP = {v: k for k, v in LITHOLOGY_MAP.items()}

# Standard Lithology Names and Colors
LITHOLOGY_LABELS = [
    'Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone',
    'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
]

LITHOLOGY_COLORS = [
    '#FFE066',  # 0: Sandstone (Bright Gold)
    '#C7F9CC',  # 1: Sandstone/Shale (Sage Green)
    '#566573',  # 2: Shale (Slate Gray)
    '#A9DFBF',  # 3: Marl (Light Muted Green)
    '#138D75',  # 4: Dolomite (Teal)
    '#5DADE2',  # 5: Limestone (Sky Blue)
    '#F4F6F7',  # 6: Chalk (Pure Off-White)
    '#EBDEF0',  # 7: Halite (Lavender Pink)
    '#D2B4DE',  # 8: Anhydrite (Soft Purple)
    '#F5CBA7',  # 9: Tuff (Apricot Orange)
    '#1C2833',  # 10: Coal (Jet Charcoal Black)
    '#922B21'   # 11: Basement (Crimson Maroon)
]

def load_force_dataset(filepath="data/raw/15_9-23.csv"):
    """
    Loads the real FORCE 2020 dataset from CSV, standardizes log names,
    handles duplicate rows, infers target lithology class indices, and
    divides the single well into 10 contiguous depth blocks (groups) to prevent spatial leakage.
    """
    logger.info(f"Loading FORCE 2020 CSV dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FORCE CSV dataset not found at {filepath}!")
        
    df = pd.read_csv(filepath)
    
    # 1. Standardize column names to uppercase and strip
    df.columns = [col.upper().strip() for col in df.columns]
    
    # 2. Handle duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < initial_len:
        logger.info(f"Dropped {initial_len - len(df)} duplicate rows.")
        
    # 3. Handle infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 4. Standardize column name variants
    col_mapping = {
        'DEPT': 'DEPTH_MD',
        'DEPTH': 'DEPTH_MD',
        'LITHOLOGY_LITHOFACIES': 'FORCE_2020_LITHOFACIES_LITHOLOGY',
        'FORCE_2020_LITHOFACIES_LITHOLOGY': 'FORCE_2020_LITHOFACIES_LITHOLOGY'
    }
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
            
    # 5. Map numeric codes to 0..11 class labels
    if 'FORCE_2020_LITHOFACIES_LITHOLOGY' in df.columns:
        df['LITHOLOGY'] = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].fillna(-1).astype(int).map(LITHOLOGY_MAP)
    else:
        logger.warning("Lithology column FORCE_2020_LITHOFACIES_LITHOLOGY not found!")
        df['LITHOLOGY'] = np.nan
        
    # Remove records without valid lithology
    df = df.dropna(subset=['LITHOLOGY']).reset_index(drop=True)
    df['LITHOLOGY'] = df['LITHOLOGY'].astype(int)
    
    # 6. Sort strictly by depth to ensure contiguous depth blocks
    df = df.sort_values(by='DEPTH_MD').reset_index(drop=True)
    
    # 7. Create 10 contiguous depth blocks as pseudo-wells
    n_samples = len(df)
    block_size = n_samples // 10
    blocks = np.clip(np.arange(n_samples) // block_size, 0, 9)
    df['WELL'] = '15/9-23_block_' + pd.Series(blocks).astype(str)
    df['WELL_ID'] = df['WELL']
    
    # Mark as real
    df['IS_SYNTHETIC'] = False
    
    logger.info(f"Successfully loaded {len(df)} real samples across 10 contiguous depth groups.")
    return df

def generate_synthetic_las_files(target_dir='data/raw', num_wells=5):
    """
    Generates geologically coherent synthetic LAS well log files to simulate
    the FORCE 2020 well log schema. Serves as training augmentations.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if files already exist
    existing_files = glob.glob(os.path.join(target_dir, "WELL_FORCE_2020_*.las"))
    if len(existing_files) >= num_wells:
        logger.info(f"Found {len(existing_files)} existing LAS files in {target_dir}. Skipping synthetic generation.")
        return
    
    logger.info(f"Generating {num_wells} synthetic training augmenter well files...")
    
    np.random.seed(42)
    
    for well_idx in range(1, num_wells + 1):
        well_name = f"WELL_FORCE_2020_{well_idx:02d}"
        filepath = os.path.join(target_dir, f"{well_name}.las")
        
        start_depth = 1000.0
        stop_depth = 2000.0
        step = 0.1
        depths = np.arange(start_depth, stop_depth + step/2.0, step)
        n_samples = len(depths)
        
        boundaries_m = np.sort(np.random.uniform(start_depth + 15, stop_depth - 15, size=17))
        boundaries_m = np.insert(boundaries_m, 0, start_depth)
        boundaries_m = np.append(boundaries_m, stop_depth)
        layer_boundaries = np.round((boundaries_m - start_depth) / step).astype(int)
        
        sedimentary_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        layer_lithologies = list(np.random.permutation(sedimentary_classes))
        
        extra_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p_extra = [0.20, 0.20, 0.35, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]
        extra_layers = list(np.random.choice(extra_classes, size=6, p=p_extra))
        layer_lithologies.extend(extra_layers)
        
        np.random.shuffle(layer_lithologies)
        layer_lithologies.append(11)  # basement bottom
        
        sample_lithologies = np.zeros(n_samples, dtype=int)
        for i in range(18):
            start_idx = int(layer_boundaries[i])
            end_idx = min(int(layer_boundaries[i+1]), n_samples)
            if start_idx < n_samples:
                sample_lithologies[start_idx:end_idx] = layer_lithologies[i]
                
        gr = np.zeros(n_samples)
        rhob = np.zeros(n_samples)
        nphi = np.zeros(n_samples)
        dtc = np.zeros(n_samples)
        pef = np.zeros(n_samples)
        sp = np.zeros(n_samples)
        rdep = np.zeros(n_samples)
        rmed = np.zeros(n_samples)
        rsha = np.zeros(n_samples)
        cali = np.zeros(n_samples)
        bs = np.full(n_samples, 8.5)
        
        for i in range(n_samples):
            lit = sample_lithologies[i]
            
            if lit == 0:  # Sandstone
                gr[i] = np.random.normal(45, 8)
                rhob[i] = np.random.normal(2.35, 0.05)
                nphi[i] = np.random.normal(0.20, 0.03)
                dtc[i] = np.random.normal(82, 5)
                pef[i] = np.random.normal(1.85, 0.1)
                sp[i] = np.random.normal(-35, 5)
                res = 10 ** np.random.normal(1.3, 0.3)
            elif lit == 1:  # Sandstone/Shale
                gr[i] = np.random.normal(75, 12)
                rhob[i] = np.random.normal(2.40, 0.06)
                nphi[i] = np.random.normal(0.26, 0.04)
                dtc[i] = np.random.normal(92, 8)
                pef[i] = np.random.normal(2.6, 0.15)
                sp[i] = np.random.normal(-15, 5)
                res = 10 ** np.random.normal(0.8, 0.25)
            elif lit == 2:  # Shale
                gr[i] = np.random.normal(115, 15)
                rhob[i] = np.random.normal(2.48, 0.05)
                nphi[i] = np.random.normal(0.36, 0.04)
                dtc[i] = np.random.normal(108, 8)
                pef[i] = np.random.normal(3.4, 0.2)
                sp[i] = np.random.normal(5, 3)
                res = 10 ** np.random.normal(0.4, 0.15)
            elif lit == 3:  # Marl
                gr[i] = np.random.normal(80, 10)
                rhob[i] = np.random.normal(2.52, 0.06)
                nphi[i] = np.random.normal(0.31, 0.03)
                dtc[i] = np.random.normal(96, 6)
                pef[i] = np.random.normal(3.2, 0.15)
                sp[i] = np.random.normal(0, 4)
                res = 10 ** np.random.normal(0.6, 0.2)
            elif lit == 4:  # Dolomite
                gr[i] = np.random.normal(25, 6)
                rhob[i] = np.random.normal(2.82, 0.04)
                nphi[i] = np.random.normal(0.06, 0.02)
                dtc[i] = np.random.normal(52, 3)
                pef[i] = np.random.normal(3.14, 0.1)
                sp[i] = np.random.normal(2, 4)
                res = 10 ** np.random.normal(1.8, 0.4)
            elif lit == 5:  # Limestone
                gr[i] = np.random.normal(20, 5)
                rhob[i] = np.random.normal(2.68, 0.03)
                nphi[i] = np.random.normal(0.10, 0.02)
                dtc[i] = np.random.normal(58, 4)
                pef[i] = np.random.normal(5.08, 0.15)
                sp[i] = np.random.normal(-5, 4)
                res = 10 ** np.random.normal(1.6, 0.3)
            elif lit == 6:  # Chalk
                gr[i] = np.random.normal(15, 4)
                rhob[i] = np.random.normal(2.28, 0.05)
                nphi[i] = np.random.normal(0.28, 0.04)
                dtc[i] = np.random.normal(74, 6)
                pef[i] = np.random.normal(4.8, 0.15)
                sp[i] = np.random.normal(-2, 3)
                res = 10 ** np.random.normal(1.1, 0.2)
            elif lit == 7:  # Halite
                gr[i] = np.random.normal(8, 2)
                rhob[i] = np.random.normal(2.08, 0.03)
                nphi[i] = np.random.normal(0.03, 0.01)
                dtc[i] = np.random.normal(67, 3)
                pef[i] = np.random.normal(4.65, 0.1)
                sp[i] = np.random.normal(8, 2)
                res = 10 ** np.random.normal(3.5, 0.5)
            elif lit == 8:  # Anhydrite
                gr[i] = np.random.normal(12, 3)
                rhob[i] = np.random.normal(2.96, 0.03)
                nphi[i] = np.random.normal(0.02, 0.01)
                dtc[i] = np.random.normal(50, 2)
                pef[i] = np.random.normal(5.0, 0.15)
                sp[i] = np.random.normal(6, 2)
                res = 10 ** np.random.normal(3.2, 0.4)
            elif lit == 9:  # Tuff
                gr[i] = np.random.normal(92, 12)
                rhob[i] = np.random.normal(2.18, 0.08)
                nphi[i] = np.random.normal(0.33, 0.05)
                dtc[i] = np.random.normal(98, 8)
                pef[i] = np.random.normal(3.0, 0.25)
                sp[i] = np.random.normal(3, 4)
                res = 10 ** np.random.normal(0.9, 0.3)
            elif lit == 10:  # Coal
                gr[i] = np.random.normal(30, 8)
                rhob[i] = np.random.normal(1.42, 0.08)
                nphi[i] = np.random.normal(0.52, 0.05)
                dtc[i] = np.random.normal(125, 12)
                pef[i] = np.random.normal(1.6, 0.15)
                sp[i] = np.random.normal(-8, 5)
                res = 10 ** np.random.normal(2.2, 0.4)
            elif lit == 11:  # Basement
                hf_noise_gr = np.random.normal(0, 15)
                hf_noise_rhob = np.random.normal(0, 0.08)
                hf_noise_nphi = np.random.normal(0, 0.003)
                hf_noise_dtc = np.random.normal(0, 4.0)
                
                gr[i] = np.random.normal(140, 22) + hf_noise_gr
                rhob[i] = np.random.normal(2.74, 0.18) + hf_noise_rhob
                nphi[i] = np.random.normal(0.015, 0.005) + hf_noise_nphi
                dtc[i] = np.random.normal(44, 6) + hf_noise_dtc
                pef[i] = np.random.normal(3.8, 0.5)
                sp[i] = np.random.normal(12, 6)
                res = 10 ** (np.random.normal(3.3, 0.7) + np.random.normal(0, 0.3))
                
            rdep[i] = res
            rmed[i] = res * np.random.normal(1.0, 0.05)
            rsha[i] = res * np.random.normal(1.0, 0.1)
            
            if lit == 2:  # Shale washout
                cali[i] = bs[i] + np.random.normal(1.4, 0.3)
            elif lit == 0:  # Sandstone mudcake
                cali[i] = bs[i] - np.random.normal(0.18, 0.04)
            else:
                cali[i] = bs[i] + np.random.normal(0.08, 0.03)

        window = 21
        def smooth(arr):
            return pd.Series(arr).rolling(window=window, min_periods=1, center=True).mean().values.copy()
            
        gr = smooth(gr)
        rhob = smooth(rhob)
        nphi = smooth(nphi)
        dtc = smooth(dtc)
        pef = smooth(pef)
        sp = smooth(sp)
        rdep = smooth(rdep)
        rmed = smooth(rmed)
        rsha = smooth(rsha)
        cali = smooth(cali)
        
        nphi = np.clip(nphi, 0.0, 0.9)
        
        nan_cols = [cali, rsha, rmed, rdep, rhob, nphi, pef, dtc, sp]
        for col in nan_cols:
            mask = np.random.rand(n_samples) < 0.001
            col[mask] = np.nan
            
        for _ in range(5):
            idx = np.random.randint(50, n_samples - 50)
            col_idx = np.random.choice(len(nan_cols))
            col_to_gap = nan_cols[col_idx]
            col_to_gap[idx:idx+3] = np.nan
            
        if well_idx == 5:
            pef[:] = np.nan
            
        las = lasio.LASFile()
        las.well['WELL'] = lasio.HeaderItem('WELL', value=well_name)
        las.well['COMP'] = lasio.HeaderItem('COMP', value='FORCE 2020 Recreator')
        
        las.insert_curve(0, 'DEPT', depths, unit='m')
        las.insert_curve(1, 'CALI', cali, unit='in')
        las.insert_curve(2, 'RSHA', rsha, unit='ohm.m')
        las.insert_curve(3, 'RMED', rmed, unit='ohm.m')
        las.insert_curve(4, 'RDEP', rdep, unit='ohm.m')
        las.insert_curve(5, 'RHOB', rhob, unit='g/cm3')
        las.insert_curve(6, 'GR', gr, unit='gAPI')
        las.insert_curve(7, 'NPHI', nphi, unit='v/v')
        las.insert_curve(8, 'PEF', pef, unit='b/e')
        las.insert_curve(9, 'DTC', dtc, unit='us/ft')
        las.insert_curve(10, 'SP', sp, unit='mV')
        las.insert_curve(11, 'BS', bs, unit='in')
        
        original_codes = np.array([REVERSE_LITHOLOGY_MAP[lit] for lit in sample_lithologies])
        las.insert_curve(12, 'FORCE_2020_LITHOFACIES_LITHOLOGY', original_codes, unit='code')
        
        las.write(filepath)
        
    logger.info(f"Generated synthetic training augmentations.")

def main():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/interim', exist_ok=True)
    
    # 1. Load real FORCE 2020 CSV dataset
    force_df = load_force_dataset('data/raw/15_9-23.csv')
    
    # 2. Make sure synthetic data exists
    generate_synthetic_las_files('data/raw', num_wells=5)
    
    # 3. Load synthetic wells and tag them
    synthetic_wells = []
    las_files = glob.glob(os.path.join('data/raw', "*.las"))
    for filepath in sorted(las_files):
        filename = os.path.basename(filepath)
        well_name = os.path.splitext(filename)[0]
        
        logger.info(f"Loading synthetic well log: {filename}")
        las = lasio.read(filepath)
        df = las.df().reset_index()
        df = df.rename(columns={'DEPT': 'DEPTH_MD', 'DEPTH': 'DEPTH_MD'})
        df.columns = [col.upper().strip() for col in df.columns]
        
        df['WELL'] = well_name
        df['WELL_ID'] = well_name
        df['IS_SYNTHETIC'] = True
        
        litho_col = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
        if litho_col in df.columns:
            df['LITHOLOGY'] = df[litho_col].fillna(-1).astype(int).map(LITHOLOGY_MAP)
            
        df = df.dropna(subset=['LITHOLOGY']).reset_index(drop=True)
        df['LITHOLOGY'] = df['LITHOLOGY'].astype(int)
        
        synthetic_wells.append(df)
        
    if synthetic_wells:
        synthetic_df = pd.concat(synthetic_wells, ignore_index=True)
        logger.info(f"Loaded {len(synthetic_df)} synthetic samples.")
        merged_df = pd.concat([force_df, synthetic_df], ignore_index=True)
    else:
        merged_df = force_df
        
    output_path = 'data/processed/merged_raw.parquet'
    merged_df.to_parquet(output_path, index=False)
    logger.info(f"Successfully saved combined parquet to {output_path} (Shape: {merged_df.shape})")

if __name__ == "__main__":
    main()
