"""
================================================================================
Transition Matrix Analysis Utility
================================================================================

Standalone script for calculating and analyzing transition matrices from
land cover classification data.

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - rasterio
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scipy

================================================================================
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.linalg import logm, expm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for transition matrix analysis."""
    
    BASE_DIR = r"D:/Publication/Habitat Loss in the Greater Kafue Ecosystem"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/transitions")
    FIGURES_DIR = os.path.join(BASE_DIR, "outputs/figures")
    TABLES_DIR = os.path.join(BASE_DIR, "outputs/tables")
    
    LC_FILES = {
        1984: os.path.join(DATA_DIR, "GKE_1984.tif"),
        1994: os.path.join(DATA_DIR, "GKE_1994.tif"),
        2004: os.path.join(DATA_DIR, "GKE_2004.tif"),
        2014: os.path.join(DATA_DIR, "GKE_2014.tif"),
        2024: os.path.join(DATA_DIR, "GKE_2024.tif"),
    }
    
    CLASS_LABELS = {
        1: 'Built-up', 2: 'Forest', 3: 'Cropland',
        4: 'Grassland', 5: 'Bareland', 6: 'Water'
    }
    
    N_CLASSES = 6
    PERIODS = [(1984, 1994), (1994, 2004), (2004, 2014), (2014, 2024), (1984, 2024)]
    NATURAL_CLASSES = [2, 4, 6]
    DISTURBED_CLASSES = [1, 3, 5]
    CMAP_TRANSITIONS = 'YlOrRd'
    FIGURE_DPI = 300


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def create_output_directories():
    """Create output directories."""
    for d in [Config.OUTPUT_DIR, Config.FIGURES_DIR, Config.TABLES_DIR]:
        os.makedirs(d, exist_ok=True)


def load_land_cover(year):
    """Load land cover raster for a given year."""
    filepath = Config.LC_FILES.get(year)
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Land cover file not found for {year}")
    
    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def calculate_transition_matrix(lc_start, lc_end, n_classes=6):
    """
    Calculate transition matrix between two land cover maps.
    
    Args:
        lc_start: Start period land cover array
        lc_end: End period land cover array
        n_classes: Number of land cover classes
        
    Returns:
        tuple: (count matrix, probability matrix)
    """
    # Align shapes
    if lc_start.shape != lc_end.shape:
        min_shape = (min(lc_start.shape[0], lc_end.shape[0]),
                     min(lc_start.shape[1], lc_end.shape[1]))
        lc_start = lc_start[:min_shape[0], :min_shape[1]]
        lc_end = lc_end[:min_shape[0], :min_shape[1]]
    
    # Count transitions
    count_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    valid = (lc_start > 0) & (lc_end > 0) & (lc_start <= n_classes) & (lc_end <= n_classes)
    
    for i in range(1, n_classes + 1):
        for j in range(1, n_classes + 1):
            count_matrix[i-1, j-1] = np.sum((lc_start == i) & (lc_end == j) & valid)
    
    # Probability matrix
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob_matrix = count_matrix / row_sums
    
    return count_matrix, prob_matrix


def annualize_matrix(prob_matrix, n_years):
    """
    Convert multi-year transition matrix to annual rates.
    
    Uses matrix logarithm method: P_annual = exp(log(P) / n)
    
    Args:
        prob_matrix: Period transition probability matrix
        n_years: Number of years in the period
        
    Returns:
        ndarray: Annual transition probability matrix
    """
    n_classes = prob_matrix.shape[0]
    
    try:
        # Add small value for numerical stability
        P_adj = prob_matrix + 1e-10
        P_adj = P_adj / P_adj.sum(axis=1, keepdims=True)
        
        log_P = logm(P_adj)
        annual_log = log_P / n_years
        annual = expm(annual_log)
        
        annual = np.real(annual)
        annual = np.clip(annual, 0, 1)
        annual = annual / annual.sum(axis=1, keepdims=True)
        
    except Exception:
        # Fallback to linear interpolation
        identity = np.eye(n_classes)
        annual = identity + (prob_matrix - identity) / n_years
        annual = np.clip(annual, 0, 1)
        annual = annual / annual.sum(axis=1, keepdims=True)
    
    return annual


def calculate_net_change(count_matrix, pixel_area_ha):
    """
    Calculate net area change for each class.
    
    Args:
        count_matrix: Transition count matrix
        pixel_area_ha: Area per pixel in hectares
        
    Returns:
        DataFrame: Net change statistics
    """
    class_names = list(Config.CLASS_LABELS.values())
    
    # Gains and losses
    gains = count_matrix.sum(axis=0) - np.diag(count_matrix)
    losses = count_matrix.sum(axis=1) - np.diag(count_matrix)
    net_change = gains - losses
    
    results = []
    for i, name in enumerate(class_names):
        results.append({
            'Class': name,
            'Gains_ha': gains[i] * pixel_area_ha,
            'Losses_ha': losses[i] * pixel_area_ha,
            'Net_Change_ha': net_change[i] * pixel_area_ha,
            'Gains_pixels': gains[i],
            'Losses_pixels': losses[i]
        })
    
    return pd.DataFrame(results)


def plot_transition_heatmap(prob_matrix, title, output_path):
    """Create publication-quality transition heatmap."""
    class_names = list(Config.CLASS_LABELS.values())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        prob_matrix * 100,
        annot=True, fmt='.2f',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap=Config.CMAP_TRANSITIONS,
        cbar_kws={'label': 'Transition Probability (%)'},
        ax=ax
    )
    
    ax.set_xlabel('To Class', fontsize=12)
    ax.set_ylabel('From Class', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def analyze_period(year_start, year_end, profile):
    """
    Analyze transitions for a single period.
    
    Args:
        year_start: Start year
        year_end: End year
        profile: Raster profile for area calculation
        
    Returns:
        dict: Analysis results
    """
    print(f"\n--- Period {year_start}-{year_end} ---")
    
    lc_start, _ = load_land_cover(year_start)
    lc_end, _ = load_land_cover(year_end)
    
    count_matrix, prob_matrix = calculate_transition_matrix(lc_start, lc_end)
    
    n_years = year_end - year_start
    annual_matrix = annualize_matrix(prob_matrix, n_years)
    
    pixel_area_ha = abs(profile['transform'][0] * profile['transform'][4]) / 10000
    net_change = calculate_net_change(count_matrix, pixel_area_ha)
    
    # Key transitions
    print("\n  Major transitions (>1%):")
    class_names = list(Config.CLASS_LABELS.values())
    for i in range(Config.N_CLASSES):
        for j in range(Config.N_CLASSES):
            if i != j and prob_matrix[i, j] > 0.01:
                print(f"    {class_names[i]} → {class_names[j]}: {prob_matrix[i, j]*100:.2f}%")
    
    # Save matrices
    period_str = f"{year_start}_{year_end}"
    
    prob_df = pd.DataFrame(prob_matrix, index=class_names, columns=class_names)
    prob_df.to_csv(os.path.join(Config.TABLES_DIR, f'transition_prob_{period_str}.csv'))
    
    annual_df = pd.DataFrame(annual_matrix, index=class_names, columns=class_names)
    annual_df.to_csv(os.path.join(Config.TABLES_DIR, f'transition_annual_{period_str}.csv'))
    
    net_change.to_csv(os.path.join(Config.TABLES_DIR, f'net_change_{period_str}.csv'), index=False)
    
    # Plot
    plot_transition_heatmap(
        prob_matrix,
        f'Land Cover Transitions {year_start}-{year_end}',
        os.path.join(Config.FIGURES_DIR, f'heatmap_transitions_{period_str}.png')
    )
    
    return {
        'period': (year_start, year_end),
        'count_matrix': count_matrix,
        'prob_matrix': prob_matrix,
        'annual_matrix': annual_matrix,
        'net_change': net_change
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("TRANSITION MATRIX ANALYSIS")
    print("Greater Kafue Ecosystem")
    print("="*60)
    
    create_output_directories()
    
    # Get reference profile
    _, profile = load_land_cover(list(Config.LC_FILES.keys())[0])
    
    # Analyze all periods
    results = {}
    for year_start, year_end in Config.PERIODS:
        try:
            results[(year_start, year_end)] = analyze_period(year_start, year_end, profile)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  Tables: {Config.TABLES_DIR}")
    print(f"  Figures: {Config.FIGURES_DIR}")
    
    return results


if __name__ == "__main__":
    results = main()
