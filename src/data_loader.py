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

def generate_synthetic_las_files(target_dir='data/raw', num_wells=10):
    """
    Generates geologically coherent synthetic LAS well log files to simulate
    the FORCE 2020 well log schema. This ensures the ML pipeline is fully runnable
    and has realistic signal relationships (e.g. Coal low density/high neutron,
    Shale high gamma, Limestone high density/low neutron, etc.)
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if files already exist
    existing_files = glob.glob(os.path.join(target_dir, "*.las"))
    if len(existing_files) >= num_wells:
        logger.info(f"Found {len(existing_files)} existing LAS files in {target_dir}. Skipping synthetic generation.")
        return
    
    logger.info(f"No raw files found. Generating {num_wells} synthetic geologically coherent LAS files...")
    
    np.random.seed(42)
    
    for well_idx in range(1, num_wells + 1):
        well_name = f"WELL_FORCE_2020_{well_idx:02d}"
        filepath = os.path.join(target_dir, f"{well_name}.las")
        
        # Define depth range: 1000m to 2000m at 0.1m interval
        start_depth = 1000.0
        stop_depth = 2000.0
        step = 0.1
        depths = np.arange(start_depth, stop_depth + step/2.0, step)
        n_samples = len(depths)
        
        # Build sedimentary layers spanning the entire well depth (exactly L = 18 layers)
        # Choose 17 random depth boundaries between start and stop depth (with a margin), sort them,
        # and convert them to sample indices to guarantee 100% depth coverage.
        boundaries_m = np.sort(np.random.uniform(start_depth + 15, stop_depth - 15, size=17))
        boundaries_m = np.insert(boundaries_m, 0, start_depth)
        boundaries_m = np.append(boundaries_m, stop_depth)
        layer_boundaries = np.round((boundaries_m - start_depth) / step).astype(int)
        
        # Assign lithology to each layer, ensuring all 12 classes appear in every single well.
        # Shuffled unique sedimentary classes (0-10) to guarantee minimum support for all classes
        sedimentary_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        layer_lithologies = list(np.random.permutation(sedimentary_classes))
        
        # Add 6 more layers drawn from common classes to maintain realistic sedimentary proportions
        extra_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p_extra = [0.20, 0.20, 0.35, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]
        extra_layers = list(np.random.choice(extra_classes, size=6, p=p_extra))
        layer_lithologies.extend(extra_layers)
        
        # Shuffle sedimentary layers for variety
        np.random.shuffle(layer_lithologies)
        
        # Geologically append Basement (11) as the very bottom layer of the well (layer index 17)
        layer_lithologies.append(11)
        
        # Map samples to lithology classes
        sample_lithologies = np.zeros(n_samples, dtype=int)
        for i in range(18):
            start_idx = int(layer_boundaries[i])
            end_idx = min(int(layer_boundaries[i+1]), n_samples)
            if start_idx < n_samples:
                sample_lithologies[start_idx:end_idx] = layer_lithologies[i]
                
        # Generate correlated logging measurements
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
        
        # Mapping physical log properties based on geological rules
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
                res = 10 ** np.random.normal(3.5, 0.5)  # Very high!
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
                rhob[i] = np.random.normal(1.42, 0.08)  # Super low density!
                nphi[i] = np.random.normal(0.52, 0.05)  # Super high neutron!
                dtc[i] = np.random.normal(125, 12)
                pef[i] = np.random.normal(1.6, 0.15)
                sp[i] = np.random.normal(-8, 5)
                res = 10 ** np.random.normal(2.2, 0.4)
            elif lit == 11:  # Basement
                # Igneous/metamorphic crystalline rocks (Granite/Gneiss) highly distinct from sedimentary Dolomite
                hf_noise_gr = np.random.normal(0, 15)
                hf_noise_rhob = np.random.normal(0, 0.08)
                hf_noise_nphi = np.random.normal(0, 0.003)
                hf_noise_dtc = np.random.normal(0, 4.0)
                
                gr[i] = np.random.normal(140, 22) + hf_noise_gr # high GR due to potassium feldspars/radioactive traces
                rhob[i] = np.random.normal(2.74, 0.18) + hf_noise_rhob # high density variance (granite to basite)
                nphi[i] = np.random.normal(0.015, 0.005) + hf_noise_nphi # very tight matrix, sharply reduced crystalline porosity
                dtc[i] = np.random.normal(44, 6) + hf_noise_dtc # low travel time / low DTC / high velocity crystalline signatures
                pef[i] = np.random.normal(3.8, 0.5)
                sp[i] = np.random.normal(12, 6)
                res = 10 ** (np.random.normal(3.3, 0.7) + np.random.normal(0, 0.3)) # high resistivity with fracture variance
                
            rdep[i] = res
            rmed[i] = res * np.random.normal(1.0, 0.05)
            rsha[i] = res * np.random.normal(1.0, 0.1)
            
            # Caliper washouts in Shale, mudcake in Sandstone
            if lit == 2:  # Shale washout
                cali[i] = bs[i] + np.random.normal(1.4, 0.3)
            elif lit == 0:  # Sandstone mudcake
                cali[i] = bs[i] - np.random.normal(0.18, 0.04)
            else:
                cali[i] = bs[i] + np.random.normal(0.08, 0.03)

        # Smooth signals with rolling average to simulate tool response/geological continuity
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
        
        # Clip neutron porosity to standard geologically sound limits [0, 1]
        nphi = np.clip(nphi, 0.0, 0.9)
        
        # Inject small gaps to test Phase 3 handler (NaN values)
        # Random isolated NaN points (0.1% chance per curve, except depth and GR)
        nan_cols = [cali, rsha, rmed, rdep, rhob, nphi, pef, dtc, sp]
        for col in nan_cols:
            mask = np.random.rand(n_samples) < 0.001
            col[mask] = np.nan
            
        # Introduce a few spots of 3 consecutive NaNs to test the small-gap median filling (< 5 samples)
        for _ in range(5):
            idx = np.random.randint(50, n_samples - 50)
            col_idx = np.random.choice(len(nan_cols))
            col_to_gap = nan_cols[col_idx]
            col_to_gap[idx:idx+3] = np.nan
            
        # For Well 05, drop PEF entirely to simulate an entirely missing log (tests row drop logic in Phase 3)
        if well_idx == 5:
            pef[:] = np.nan
            logger.info("Well 5 PEF curve set to all NaN to simulate an entirely missing log.")
            
        # Create a lasio LAS file
        las = lasio.LASFile()
        las.well['WELL'] = lasio.HeaderItem('WELL', value=well_name)
        las.well['COMP'] = lasio.HeaderItem('COMP', value='FORCE 2020 Recreator')
        
        # Insert curves
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
        
        # Add lithofacies using the original FORCE 2020 integer codes (e.g. 30000, 65000)
        original_codes = np.array([REVERSE_LITHOLOGY_MAP[lit] for lit in sample_lithologies])
        las.insert_curve(12, 'FORCE_2020_LITHOFACIES_LITHOLOGY', original_codes, unit='code')
        
        # Write out
        las.write(filepath)
        
    logger.info(f"Synthetic well LAS files successfully generated in {target_dir}!")

def load_las_dataset(data_dir='data/raw'):
    """
    Loads all LAS files in data_dir, extracts curves, merges them,
    standardizes column names, and maps geological lithofacies codes to integer classes (0-11).
    """
    las_files = glob.glob(os.path.join(data_dir, "*.las"))
    if not las_files:
        raise FileNotFoundError(f"No LAS files found in {data_dir}!")
        
    logger.info(f"Loading and merging {len(las_files)} LAS files from {data_dir}...")
    
    all_wells_data = []
    
    for filepath in sorted(las_files):
        filename = os.path.basename(filepath)
        well_id = os.path.splitext(filename)[0]
        
        logger.info(f"Reading LAS file: {filename}")
        las = lasio.read(filepath)
        df = las.df()
        
        # In lasio, depth is the index. Let's make it a column called 'DEPTH_MD'
        df = df.reset_index()
        df = df.rename(columns={'DEPT': 'DEPTH_MD', 'DEPTH': 'DEPTH_MD'})
        
        # Standardize column names to uppercase and strip
        df.columns = [col.upper().strip() for col in df.columns]
        
        # Add well identifiers
        df['WELL_ID'] = well_id
        
        # Map the force lithology column to LITHOLOGY
        litho_col = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
        if litho_col in df.columns:
            df['LITHOLOGY'] = df[litho_col].map(LITHOLOGY_MAP)
        else:
            logger.warning(f"Lithofacies column {litho_col} not found in {filename}!")
            
        all_wells_data.append(df)
        
    # Concatenate all wells into a single DataFrame
    merged_df = pd.concat(all_wells_data, ignore_index=True)
    logger.info(f"Merged DataFrame shape: {merged_df.shape}")
    
    return merged_df

def main():
    # Make sure all directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/interim', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('src', exist_ok=True)
    
    # Generate synthetic well data if directory is empty
    generate_synthetic_las_files('data/raw', num_wells=10)
    
    # Load and merge into processed Parquet format
    merged_df = load_las_dataset('data/raw')
    output_path = 'data/processed/merged_raw.parquet'
    
    # Save as Parquet
    merged_df.to_parquet(output_path, index=False)
    logger.info(f"Merged well data successfully saved to {output_path}!")

if __name__ == "__main__":
    main()
