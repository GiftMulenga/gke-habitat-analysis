"""
================================================================================
Susceptibility Mapping and CA-Markov Projections
================================================================================

Objective B/C: Habitat Loss Drivers and Future Projections - Greater Kafue Ecosystem

This script generates:
    - Habitat loss susceptibility maps from Random Forest probabilities
    - Transition matrices for CA-Markov modeling
    - Future land cover projections under multiple scenarios

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - rasterio
    - numpy
    - pandas
    - matplotlib
    - scikit-learn
    - joblib
    - scipy
    - geopandas

================================================================================
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import joblib
from scipy import ndimage
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for susceptibility mapping and CA-Markov projections."""
    
    # Directory Configuration - UPDATE THESE PATHS
    BASE_DIR = r"E:/Research/Msc_Tropical_Ecology/GKE_Objective_B"
    LC_DIR = r"D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/data"
    DRIVER_BASE_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "results")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    TABLES_DIR = os.path.join(BASE_DIR, "tables")
    MAP_DIR = os.path.join(BASE_DIR, "maps")
    PROJECTION_DIR = os.path.join(BASE_DIR, "projections")
    
    # Land cover files
    LC_FILES = {
        1984: os.path.join(LC_DIR, "GKE_1984.tif"),
        1994: os.path.join(LC_DIR, "GKE_1994.tif"),
        2004: os.path.join(LC_DIR, "GKE_2004.tif"),
        2014: os.path.join(LC_DIR, "GKE_2014.tif"),
        2024: os.path.join(LC_DIR, "GKE_2024.tif"),
    }
    
    # Driver files (14 variables)
    DRIVER_FILES = {
        'dist_roads': ('proximity', 'dist_roads.tif'),
        'dist_settlements': ('proximity', 'dist_settlements.tif'),
        'dist_rivers': ('proximity', 'dist_rivers.tif'),
        'dist_knp': ('proximity', 'dist_knp.tif'),
        'pop_density': ('socioeconomic', 'pop_density.tif'),
        'pop_change': ('socioeconomic', 'pop_change.tif'),
        'pct_cultivated': ('socioeconomic', 'pct_cultivated.tif'),
        'protection_status': ('conservation', 'protection_status.tif'),
        'elevation': ('topographic', 'elevation.tif'),
        'slope': ('topographic', 'slope.tif'),
        'aspect': ('topographic', 'aspect.tif'),
        'twi': ('topographic', 'twi.tif'),
        'mean_rainfall': ('climatic', 'mean_rainfall.tif'),
        'mean_temp': ('climatic', 'mean_temp.tif')
    }
    
    # Feature order for model
    DRIVER_ORDER = [
        'dist_roads', 'dist_settlements', 'dist_rivers', 'dist_knp',
        'pop_density', 'pop_change', 'pct_cultivated',
        'protection_status',
        'elevation', 'slope', 'aspect', 'twi',
        'mean_rainfall', 'mean_temp'
    ]
    
    # Land cover classes
    LC_CLASSES = {
        1: 'Built-up', 2: 'Forest', 3: 'Cropland',
        4: 'Grassland', 5: 'Bareland', 6: 'Water'
    }
    
    N_CLASSES = 6
    
    # Susceptibility classification
    SUSCEPTIBILITY_BREAKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    SUSCEPTIBILITY_LABELS = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    SUSCEPTIBILITY_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8b0000']
    
    # CA-Markov parameters
    CA_NEIGHBORHOOD_SIZE = 5
    CA_WEIGHT_CONTIGUITY = 0.5
    N_ENSEMBLE_RUNS = 10
    PROJECTION_YEARS = [2030, 2040, 2050]
    
    # Processing
    BATCH_SIZE = 100000
    RANDOM_STATE = 42


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_output_directories():
    """Create output directory structure."""
    dirs = [Config.OUTPUT_DIR, Config.FIGURES_DIR, Config.TABLES_DIR,
            Config.MAP_DIR, Config.PROJECTION_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Output directories created")


def load_raster(filepath):
    """Load raster and return data with profile."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
    return data, profile, transform


def save_raster(data, profile, filepath):
    """Save array as GeoTIFF."""
    profile.update(dtype=data.dtype, count=1)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data, 1)


# =============================================================================
# SUSCEPTIBILITY MAPPING
# =============================================================================

def load_driver_stack(reference_shape):
    """
    Load and stack driver variables.
    
    Args:
        reference_shape: Target shape for alignment
        
    Returns:
        tuple: (stacked array, valid mask, profile)
    """
    print("\n--- Loading Driver Variables ---")
    
    drivers = []
    profile = None
    
    for var_name in Config.DRIVER_ORDER:
        subfolder, filename = Config.DRIVER_FILES[var_name]
        filepath = os.path.join(Config.DRIVER_BASE_DIR, subfolder, filename)
        
        if not os.path.exists(filepath):
            print(f"  ⚠ Missing: {var_name}")
            continue
        
        with rasterio.open(filepath) as src:
            data = src.read(1)
            if profile is None:
                profile = src.profile.copy()
            
            # Align if needed
            if data.shape != reference_shape:
                data = data[:reference_shape[0], :reference_shape[1]]
            
            drivers.append(data)
            print(f"  ✓ {var_name}: {data.shape}")
    
    # Stack
    driver_stack = np.stack(drivers, axis=-1)
    
    # Create valid mask
    valid_mask = ~np.any(np.isnan(driver_stack), axis=-1)
    
    print(f"\n  Driver stack shape: {driver_stack.shape}")
    print(f"  Valid pixels: {np.sum(valid_mask):,}")
    
    return driver_stack, valid_mask, profile


def generate_susceptibility_map(model_path, driver_stack, valid_mask, profile):
    """
    Generate susceptibility map from Random Forest model.
    
    Args:
        model_path: Path to trained RF model
        driver_stack: Stacked driver variables
        valid_mask: Valid data mask
        profile: Raster profile
        
    Returns:
        tuple: (probability map, class map)
    """
    print("\n--- Generating Susceptibility Map ---")
    
    # Load model
    model = joblib.load(model_path)
    print(f"  Model loaded: {model_path}")
    
    # Prepare output
    shape = driver_stack.shape[:2]
    prob_map = np.full(shape, np.nan, dtype=np.float32)
    
    # Get valid pixels
    valid_idx = np.where(valid_mask)
    n_pixels = len(valid_idx[0])
    
    print(f"  Processing {n_pixels:,} pixels...")
    
    # Process in batches
    for i in range(0, n_pixels, Config.BATCH_SIZE):
        batch_end = min(i + Config.BATCH_SIZE, n_pixels)
        batch_rows = valid_idx[0][i:batch_end]
        batch_cols = valid_idx[1][i:batch_end]
        
        # Extract features
        X_batch = driver_stack[batch_rows, batch_cols, :]
        
        # Handle NaN
        X_batch = np.nan_to_num(X_batch, nan=0)
        
        # Predict
        proba = model.predict_proba(X_batch)[:, 1]
        
        # Store
        prob_map[batch_rows, batch_cols] = proba
        
        if (i // Config.BATCH_SIZE) % 100 == 0:
            print(f"    Processed {batch_end:,}/{n_pixels:,} pixels")
    
    # Classify
    class_map = np.digitize(prob_map, Config.SUSCEPTIBILITY_BREAKS[1:]) + 1
    class_map = np.where(np.isnan(prob_map), 0, class_map).astype(np.uint8)
    
    # Statistics
    print(f"\n  Susceptibility statistics:")
    print(f"    Min:  {np.nanmin(prob_map):.4f}")
    print(f"    Max:  {np.nanmax(prob_map):.4f}")
    print(f"    Mean: {np.nanmean(prob_map):.4f}")
    
    # Save maps
    save_raster(prob_map, profile, os.path.join(Config.MAP_DIR, 'susceptibility_probability.tif'))
    print("  ✓ Saved: susceptibility_probability.tif")
    
    save_raster(class_map, profile, os.path.join(Config.MAP_DIR, 'susceptibility_class.tif'))
    print("  ✓ Saved: susceptibility_class.tif")
    
    return prob_map, class_map


# =============================================================================
# TRANSITION MATRIX
# =============================================================================

def calculate_transition_matrix(lc_start, lc_end, n_classes=6):
    """
    Calculate transition probability matrix.
    
    Args:
        lc_start: Start period land cover
        lc_end: End period land cover
        n_classes: Number of land cover classes
        
    Returns:
        ndarray: Transition probability matrix
    """
    # Align shapes
    if lc_start.shape != lc_end.shape:
        min_shape = (min(lc_start.shape[0], lc_end.shape[0]),
                     min(lc_start.shape[1], lc_end.shape[1]))
        lc_start = lc_start[:min_shape[0], :min_shape[1]]
        lc_end = lc_end[:min_shape[0], :min_shape[1]]
    
    # Count transitions
    trans_counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    valid_mask = (lc_start > 0) & (lc_end > 0) & (lc_start <= n_classes) & (lc_end <= n_classes)
    
    for i in range(1, n_classes + 1):
        for j in range(1, n_classes + 1):
            trans_counts[i-1, j-1] = np.sum((lc_start == i) & (lc_end == j) & valid_mask)
    
    # Convert to probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_probs = trans_counts / row_sums
    
    return trans_probs


def annualize_matrix(trans_matrix, n_years):
    """
    Convert multi-year transition matrix to annual rates.
    
    Args:
        trans_matrix: Period transition matrix
        n_years: Number of years in period
        
    Returns:
        ndarray: Annual transition matrix
    """
    from scipy.linalg import logm, expm
    
    n_classes = trans_matrix.shape[0]
    
    try:
        P_adjusted = trans_matrix + 1e-10
        P_adjusted = P_adjusted / P_adjusted.sum(axis=1, keepdims=True)
        
        log_P = logm(P_adjusted)
        annual_log = log_P / n_years
        annual_matrix = expm(annual_log)
        
        annual_matrix = np.real(annual_matrix)
        annual_matrix = np.clip(annual_matrix, 0, 1)
        annual_matrix = annual_matrix / annual_matrix.sum(axis=1, keepdims=True)
        
    except Exception:
        # Fallback to linear approximation
        identity = np.eye(n_classes)
        annual_matrix = identity + (trans_matrix - identity) / n_years
        annual_matrix = np.clip(annual_matrix, 0, 1)
        annual_matrix = annual_matrix / annual_matrix.sum(axis=1, keepdims=True)
    
    return annual_matrix


# =============================================================================
# CA-MARKOV SIMULATION
# =============================================================================

def ca_markov_step(current_lc, trans_matrix, susceptibility, knp_mask=None, 
                   w_contiguity=0.5, random_state=None):
    """
    Perform one step of CA-Markov simulation.
    
    Args:
        current_lc: Current land cover array
        trans_matrix: Annual transition matrix
        susceptibility: Susceptibility map
        knp_mask: Protected area mask
        w_contiguity: Weight for neighborhood contiguity
        random_state: Random seed
        
    Returns:
        ndarray: Updated land cover
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    shape = current_lc.shape
    new_lc = current_lc.copy()
    
    # Create neighborhood kernel
    kernel_size = Config.CA_NEIGHBORHOOD_SIZE
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    # Calculate neighborhood composition for each class
    neighborhood_probs = {}
    for c in range(1, Config.N_CLASSES + 1):
        class_mask = (current_lc == c).astype(float)
        neighborhood_probs[c] = ndimage.convolve(class_mask, kernel, mode='constant')
    
    # Process each pixel
    valid_mask = (current_lc > 0) & (current_lc <= Config.N_CLASSES)
    valid_idx = np.where(valid_mask)
    
    for i in range(len(valid_idx[0])):
        row, col = valid_idx[0][i], valid_idx[1][i]
        current_class = current_lc[row, col]
        
        # Skip protected areas
        if knp_mask is not None and knp_mask[row, col]:
            continue
        
        # Get base transition probabilities
        trans_probs = trans_matrix[current_class - 1].copy()
        
        # Modify by susceptibility
        susc = susceptibility[row, col] if not np.isnan(susceptibility[row, col]) else 0.5
        
        # Modify by neighborhood
        for c in range(1, Config.N_CLASSES + 1):
            neigh_prob = neighborhood_probs[c][row, col]
            trans_probs[c-1] = (1 - w_contiguity) * trans_probs[c-1] + w_contiguity * neigh_prob
        
        # Normalize
        trans_probs = trans_probs / trans_probs.sum()
        
        # Sample new class
        new_class = np.random.choice(range(1, Config.N_CLASSES + 1), p=trans_probs)
        new_lc[row, col] = new_class
    
    return new_lc


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(run_susceptibility=True, run_projections=False):
    """Main execution function."""
    print("="*60)
    print("SUSCEPTIBILITY MAPPING AND CA-MARKOV PROJECTIONS")
    print("Greater Kafue Ecosystem")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    create_output_directories()
    
    # Load reference land cover
    print("\n--- Loading Reference Data ---")
    lc_2024, profile, transform = load_raster(Config.LC_FILES[2024])
    print(f"  Reference shape: {lc_2024.shape}")
    
    results = {}
    
    if run_susceptibility:
        # Load drivers
        driver_stack, valid_mask, _ = load_driver_stack(lc_2024.shape)
        
        # Generate susceptibility
        model_path = os.path.join(Config.MODEL_DIR, 'rf_model.joblib')
        prob_map, class_map = generate_susceptibility_map(
            model_path, driver_stack, valid_mask, profile
        )
        
        # Calculate area statistics
        print("\n" + "="*60)
        print("SUSCEPTIBILITY STATISTICS")
        print("="*60)
        
        pixel_area_ha = abs(profile['transform'][0] * profile['transform'][4]) / 10000
        
        stats = []
        for i, label in enumerate(Config.SUSCEPTIBILITY_LABELS, 1):
            count = np.sum(class_map == i)
            area = count * pixel_area_ha
            pct = count / np.sum(class_map > 0) * 100 if np.sum(class_map > 0) > 0 else 0
            stats.append({'Class': label, 'Area_ha': area, 'Percent': pct})
            print(f"  {label}: {area:,.0f} ha ({pct:.1f}%)")
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Susceptibility_Area.csv'), index=False)
        
        results['susceptibility'] = {'prob_map': prob_map, 'class_map': class_map}
    
    if run_projections:
        # Load land cover maps
        print("\n--- Loading Land Cover Maps ---")
        lc_1984, _, _ = load_raster(Config.LC_FILES[1984])
        lc_2024, _, _ = load_raster(Config.LC_FILES[2024])
        
        # Calculate transition matrix
        print("\n--- Calculating Transition Matrix ---")
        trans_40yr = calculate_transition_matrix(lc_1984, lc_2024)
        trans_annual = annualize_matrix(trans_40yr, 40)
        
        # Save transition matrices
        class_names = list(Config.LC_CLASSES.values())
        trans_df = pd.DataFrame(trans_annual, index=class_names, columns=class_names)
        trans_df.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Transition_Matrix_Annual.csv'))
        
        print("\n  Key annual transitions:")
        for i, from_class in enumerate(class_names):
            for j, to_class in enumerate(class_names):
                if i != j and trans_annual[i, j] > 0.003:
                    print(f"    {from_class} → {to_class}: {trans_annual[i, j]*100:.3f}%")
        
        results['transition_matrix'] = trans_annual
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Maps: {Config.MAP_DIR}")
    print(f"  Tables: {Config.TABLES_DIR}")
    
    return results


if __name__ == "__main__":
    results = main(run_susceptibility=True, run_projections=True)
