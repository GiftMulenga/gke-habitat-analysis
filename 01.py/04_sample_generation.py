"""
================================================================================
Sample Generation and Data Extraction
================================================================================

Objective B: Drivers of Habitat Loss Analysis - Greater Kafue Ecosystem

This script generates stratified random samples for Random Forest analysis:
    - Creates habitat loss mask from LULC rasters
    - Generates spatially stratified samples
    - Extracts driver variable values
    - Performs multicollinearity analysis
    - Applies spatial train-test split

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - geopandas
    - rasterio
    - numpy
    - pandas
    - shapely
    - scipy
    - scikit-learn

================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for sample generation."""
    
    # Directory paths - UPDATE THESE
    LULC_DIR = r"D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/data"
    DRIVER_DIR = r"E:/Research/Msc_Tropical_Ecology/GKE_Objective_B/data"
    OUTPUT_DIR = r"D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_b"
    
    # Land cover class definitions
    NATURAL_CLASSES = [2, 4, 6]  # Forest, Grassland, Water
    DISTURBED_CLASSES = [1, 3, 5]  # Built-up, Cropland, Bareland
    
    CLASS_NAMES = {
        1: 'Built-up', 2: 'Forest', 3: 'Cropland',
        4: 'Grassland', 5: 'Bareland', 6: 'Water'
    }
    
    # Sampling parameters
    TOTAL_SAMPLES = 10000
    LOSS_SAMPLES = 5000
    NO_LOSS_SAMPLES = 5000
    MIN_DISTANCE = 500  # meters
    RANDOM_STATE = 42
    TEST_SIZE = 0.20
    
    # Spatial blocking
    USE_SPATIAL_BLOCKING = True
    N_SPATIAL_BLOCKS = 25
    
    # Driver variable files
    DRIVER_FILES = {
        'dist_roads': 'proximity/dist_roads.tif',
        'dist_settlements': 'proximity/dist_settlements.tif',
        'dist_rivers': 'proximity/dist_rivers.tif',
        'dist_knp': 'proximity/dist_knp.tif',
        'pop_density': 'socioeconomic/pop_density.tif',
        'pop_change': 'socioeconomic/pop_change.tif',
        'pct_cultivated': 'socioeconomic/pct_cultivated.tif',
        'protection_status': 'conservation/protection_status.tif',
        'years_protected': 'conservation/years_protected.tif',
        'elevation': 'topographic/elevation.tif',
        'slope': 'topographic/slope.tif',
        'aspect': 'topographic/aspect.tif',
        'twi': 'topographic/twi.tif',
        'mean_rainfall': 'climatic/mean_rainfall.tif',
        'mean_temp': 'climatic/mean_temp.tif',
    }
    
    # Variables to exclude (VIF analysis)
    VARS_TO_DROP_VIF = ['years_protected']
    
    # Variable categories
    VARIABLE_CATEGORIES = {
        'dist_roads': 'Proximity', 'dist_settlements': 'Proximity',
        'dist_rivers': 'Proximity', 'dist_knp': 'Proximity',
        'pop_density': 'Socio-economic', 'pop_change': 'Socio-economic',
        'pct_cultivated': 'Socio-economic', 'protection_status': 'Conservation',
        'years_protected': 'Conservation', 'elevation': 'Topographic',
        'slope': 'Topographic', 'aspect': 'Topographic', 'twi': 'Topographic',
        'mean_rainfall': 'Climatic', 'mean_temp': 'Climatic',
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_output_directories():
    """Create output directory structure."""
    dirs = [
        Config.OUTPUT_DIR,
        os.path.join(Config.OUTPUT_DIR, 'samples'),
        os.path.join(Config.OUTPUT_DIR, 'metadata'),
        os.path.join(Config.OUTPUT_DIR, 'figures'),
        os.path.join(Config.OUTPUT_DIR, 'tables')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Output directories ready")


def align_raster_to_reference(source_path, reference_path, output_path):
    """Align a raster to a reference raster's extent and resolution."""
    with rasterio.open(reference_path) as ref:
        ref_transform, ref_crs, ref_shape = ref.transform, ref.crs, ref.shape
        
    with rasterio.open(source_path) as src:
        dst_array = np.empty(ref_shape, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest
        )
        
        profile = src.profile.copy()
        profile.update({
            'height': ref_shape[0],
            'width': ref_shape[1],
            'transform': ref_transform,
            'crs': ref_crs
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_array, 1)
    
    return output_path


# =============================================================================
# HABITAT LOSS MASK
# =============================================================================

def create_habitat_loss_mask():
    """
    Create binary habitat loss mask from LULC rasters.
    
    Loss defined as: Natural habitat in 1984 → Disturbed in 2024
    
    Returns:
        dict: Mask data, profile, and statistics
    """
    print("\n" + "="*60)
    print("CREATING HABITAT LOSS MASK")
    print("="*60)
    
    lulc_1984_path = os.path.join(Config.LULC_DIR, "GKE_1984.tif")
    lulc_2024_path = os.path.join(Config.LULC_DIR, "GKE_2024.tif")
    
    with rasterio.open(lulc_1984_path) as src_1984:
        lulc_1984 = src_1984.read(1)
        profile = src_1984.profile.copy()
        transform = src_1984.transform
        crs = src_1984.crs
        res = src_1984.res[0]
        
    with rasterio.open(lulc_2024_path) as src_2024:
        lulc_2024 = src_2024.read(1)
    
    # Align if needed
    if lulc_1984.shape != lulc_2024.shape:
        print(f"  ⚠ Dimension mismatch - aligning...")
        aligned_path = os.path.join(Config.OUTPUT_DIR, "GKE_2024_aligned.tif")
        align_raster_to_reference(lulc_2024_path, lulc_1984_path, aligned_path)
        with rasterio.open(aligned_path) as src:
            lulc_2024 = src.read(1)
    
    print(f"\n  Natural classes: {Config.NATURAL_CLASSES}")
    print(f"  Disturbed classes: {Config.DISTURBED_CLASSES}")
    
    # Create masks
    natural_1984 = np.isin(lulc_1984, Config.NATURAL_CLASSES)
    disturbed_2024 = np.isin(lulc_2024, Config.DISTURBED_CLASSES)
    
    # Loss mask: Natural in 1984 AND Disturbed in 2024
    loss_mask = natural_1984 & disturbed_2024
    
    # No-loss mask: Natural in 1984 AND Natural in 2024
    natural_2024 = np.isin(lulc_2024, Config.NATURAL_CLASSES)
    no_loss_mask = natural_1984 & natural_2024
    
    # Statistics
    n_natural_1984 = np.sum(natural_1984)
    n_loss = np.sum(loss_mask)
    n_no_loss = np.sum(no_loss_mask)
    loss_rate = n_loss / n_natural_1984 * 100 if n_natural_1984 > 0 else 0
    
    print(f"\n  Natural pixels 1984: {n_natural_1984:,}")
    print(f"  Loss pixels: {n_loss:,}")
    print(f"  No-loss pixels: {n_no_loss:,}")
    print(f"  Loss rate: {loss_rate:.2f}%")
    
    # Save mask
    profile.update(dtype=np.uint8, count=1)
    mask_path = os.path.join(Config.OUTPUT_DIR, "habitat_loss_mask.tif")
    with rasterio.open(mask_path, 'w', **profile) as dst:
        dst.write(loss_mask.astype(np.uint8), 1)
    print(f"  ✓ Loss mask saved: {mask_path}")
    
    return {
        'loss_mask': loss_mask,
        'no_loss_mask': no_loss_mask,
        'profile': profile,
        'transform': transform,
        'crs': crs,
        'resolution': res,
        'shape': lulc_1984.shape,
        'stats': {
            'n_natural_1984': n_natural_1984,
            'n_loss': n_loss,
            'n_no_loss': n_no_loss,
            'loss_rate': loss_rate
        }
    }


# =============================================================================
# SAMPLE GENERATION
# =============================================================================

def generate_stratified_samples(mask_data):
    """
    Generate stratified random samples with minimum distance constraint.
    
    Args:
        mask_data: Dictionary from create_habitat_loss_mask
        
    Returns:
        dict: Sample coordinates, labels, and metadata
    """
    print("\n" + "="*60)
    print("GENERATING STRATIFIED RANDOM SAMPLES")
    print("="*60)
    
    loss_mask = mask_data['loss_mask']
    no_loss_mask = mask_data['no_loss_mask']
    transform = mask_data['transform']
    
    np.random.seed(Config.RANDOM_STATE)
    
    # Get indices for each class
    loss_indices = np.where(loss_mask)
    no_loss_indices = np.where(no_loss_mask)
    
    # Random sample indices
    n_loss = min(Config.LOSS_SAMPLES, len(loss_indices[0]))
    n_no_loss = min(Config.NO_LOSS_SAMPLES, len(no_loss_indices[0]))
    
    loss_sample_idx = np.random.choice(len(loss_indices[0]), n_loss, replace=False)
    no_loss_sample_idx = np.random.choice(len(no_loss_indices[0]), n_no_loss, replace=False)
    
    # Get coordinates
    loss_rows = loss_indices[0][loss_sample_idx]
    loss_cols = loss_indices[1][loss_sample_idx]
    no_loss_rows = no_loss_indices[0][no_loss_sample_idx]
    no_loss_cols = no_loss_indices[1][no_loss_sample_idx]
    
    # Combine
    all_rows = np.concatenate([loss_rows, no_loss_rows])
    all_cols = np.concatenate([loss_cols, no_loss_cols])
    all_labels = np.concatenate([np.ones(n_loss), np.zeros(n_no_loss)])
    
    # Convert to coordinates
    xs = transform[2] + all_cols * transform[0] + transform[0] / 2
    ys = transform[5] + all_rows * transform[4] + transform[4] / 2
    
    # Apply minimum distance filter
    print(f"\n--- Applying {Config.MIN_DISTANCE}m Minimum Distance Filter ---")
    coords = np.column_stack([xs, ys])
    
    selected = [0]
    tree = cKDTree(coords[selected])
    
    for i in range(1, len(coords)):
        dist, _ = tree.query(coords[i])
        if dist >= Config.MIN_DISTANCE:
            selected.append(i)
            tree = cKDTree(coords[selected])
    
    print(f"  Before: {len(coords)}, After: {len(selected)}")
    
    return {
        'x': xs[selected],
        'y': ys[selected],
        'row': all_rows[selected],
        'col': all_cols[selected],
        'habitat_loss': all_labels[selected].astype(int),
        'n_samples': len(selected)
    }


# =============================================================================
# DRIVER VALUE EXTRACTION
# =============================================================================

def extract_driver_values(sample_data):
    """
    Extract driver variable values at sample locations.
    
    Args:
        sample_data: Dictionary from generate_stratified_samples
        
    Returns:
        DataFrame: Samples with all driver values
    """
    print("\n" + "="*60)
    print("EXTRACTING DRIVER VALUES")
    print("="*60)
    
    df = pd.DataFrame({
        'x': sample_data['x'],
        'y': sample_data['y'],
        'row': sample_data['row'],
        'col': sample_data['col'],
        'habitat_loss': sample_data['habitat_loss']
    })
    
    for var_name, rel_path in Config.DRIVER_FILES.items():
        raster_path = os.path.join(Config.DRIVER_DIR, rel_path)
        
        if not os.path.exists(raster_path):
            print(f"  ⚠ {var_name}: File not found")
            continue
        
        try:
            with rasterio.open(raster_path) as src:
                values = []
                for x, y in zip(df['x'], df['y']):
                    row, col = src.index(x, y)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        val = src.read(1)[row, col]
                        values.append(val if val != src.nodata else np.nan)
                    else:
                        values.append(np.nan)
                
                df[var_name] = values
                valid = df[var_name].notna().sum()
                print(f"  ✓ {var_name}: {valid}/{len(df)} valid")
                
        except Exception as e:
            print(f"  ✗ {var_name}: {e}")
    
    # Remove rows with NaN
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f"\n  Removed {n_before - n_after} rows with NoData")
    print(f"  Final samples: {n_after}")
    
    return df


# =============================================================================
# MULTICOLLINEARITY ANALYSIS
# =============================================================================

def calculate_vif(df, features):
    """
    Calculate Variance Inflation Factor for each feature.
    
    Args:
        df: DataFrame with features
        features: List of feature names
        
    Returns:
        DataFrame: VIF values for each feature
    """
    vif_data = []
    
    for i, feature in enumerate(features):
        X = df[features].drop(columns=[feature]).values
        y = df[feature].values
        
        if np.std(y) < 1e-10:
            vif_data.append({'Variable': feature, 'VIF': np.inf})
            continue
        
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vif_data.append({'Variable': feature, 'VIF': vif})
    
    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


def analyze_multicollinearity(df, features):
    """
    Analyze multicollinearity using correlation and VIF.
    
    Args:
        df: DataFrame with features
        features: List of feature names
        
    Returns:
        DataFrame: VIF analysis results
    """
    print("\n--- Multicollinearity Analysis ---")
    
    # Correlation analysis
    corr_matrix = df[features].corr()
    
    print("\n  Highly correlated pairs (|r| > 0.7):")
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append((features[i], features[j], corr))
                print(f"    {features[i]} - {features[j]}: r = {corr:.2f}")
    
    if not high_corr_pairs:
        print("    None found")
    
    # VIF analysis
    vif_df = calculate_vif(df, features)
    
    print("\n  Variance Inflation Factors:")
    for _, row in vif_df.iterrows():
        print(f"    {row['Variable']}: {row['VIF']:.2f}")
    
    print(f"\n  Variables to drop: {Config.VARS_TO_DROP_VIF}")
    
    return vif_df


# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

def spatial_train_test_split(df, test_size, n_blocks, random_state):
    """
    Perform spatially-blocked train-test split.
    
    Args:
        df: DataFrame with x, y coordinates
        test_size: Proportion for test set
        n_blocks: Number of spatial blocks
        random_state: Random seed
        
    Returns:
        tuple: (train_df, test_df)
    """
    print("\n--- Spatial Train-Test Split ---")
    
    # Create spatial blocks
    x_bins = pd.cut(df['x'], bins=int(np.sqrt(n_blocks)), labels=False)
    y_bins = pd.cut(df['y'], bins=int(np.sqrt(n_blocks)), labels=False)
    df['spatial_block'] = x_bins * int(np.sqrt(n_blocks)) + y_bins
    
    # Get unique blocks
    unique_blocks = df['spatial_block'].unique()
    n_test_blocks = max(1, int(len(unique_blocks) * test_size))
    
    np.random.seed(random_state)
    test_blocks = np.random.choice(unique_blocks, n_test_blocks, replace=False)
    
    train_df = df[~df['spatial_block'].isin(test_blocks)].copy()
    test_df = df[df['spatial_block'].isin(test_blocks)].copy()
    
    print(f"  Spatial blocks: {len(unique_blocks)}")
    print(f"  Test blocks: {len(test_blocks)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df


def standard_train_test_split(df, test_size, random_state):
    """Perform standard random train-test split."""
    print("\n--- Standard Train-Test Split ---")
    
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df['habitat_loss']
    )
    
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, test_df


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_outputs(df, train_df, test_df, features, vif_df, sample_data):
    """Save all outputs."""
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    # Sample points shapefile
    geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32735")
    gdf.to_file(os.path.join(Config.OUTPUT_DIR, 'samples', 'sample_points.shp'))
    print("  ✓ Sample points shapefile saved")
    
    # Train/test CSVs
    train_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'samples', 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'samples', 'test_data.csv'), index=False)
    print("  ✓ Train/test CSVs saved")
    
    # VIF results
    vif_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'tables', 'vif_analysis.csv'), index=False)
    
    # Metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'features': features,
        'dropped_vars': Config.VARS_TO_DROP_VIF,
        'spatial_blocking': Config.USE_SPATIAL_BLOCKING,
        'random_state': Config.RANDOM_STATE,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, 'metadata', 'sampling_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Metadata saved")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("SAMPLE GENERATION")
    print("Greater Kafue Ecosystem - Habitat Loss Analysis")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  - Natural classes: {Config.NATURAL_CLASSES}")
    print(f"  - Spatial blocking: {Config.USE_SPATIAL_BLOCKING}")
    print(f"  - Variables to drop (VIF): {Config.VARS_TO_DROP_VIF}")
    
    create_output_directories()
    
    # Create loss mask
    mask_data = create_habitat_loss_mask()
    
    # Generate samples
    sample_data = generate_stratified_samples(mask_data)
    
    # Extract driver values
    df = extract_driver_values(sample_data)
    
    # Define features
    all_features = [f for f in Config.DRIVER_FILES.keys() if f in df.columns]
    
    # VIF analysis
    vif_df = analyze_multicollinearity(df, all_features)
    
    # Drop high-VIF variables
    features = [f for f in all_features if f not in Config.VARS_TO_DROP_VIF]
    print(f"\n  Features after VIF filtering: {len(features)}")
    
    # Train-test split
    if Config.USE_SPATIAL_BLOCKING:
        train_df, test_df = spatial_train_test_split(
            df, Config.TEST_SIZE, Config.N_SPATIAL_BLOCKS, Config.RANDOM_STATE
        )
    else:
        train_df, test_df = standard_train_test_split(
            df, Config.TEST_SIZE, Config.RANDOM_STATE
        )
    
    # Save outputs
    save_outputs(df, train_df, test_df, features, vif_df, sample_data)
    
    print("\n" + "="*60)
    print("SAMPLE GENERATION COMPLETE")
    print("="*60)
    print(f"  Total samples: {len(df)}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Features: {len(features)} (dropped: {Config.VARS_TO_DROP_VIF})")
    
    return {'df': df, 'train_df': train_df, 'test_df': test_df, 'features': features}


if __name__ == "__main__":
    results = main()
