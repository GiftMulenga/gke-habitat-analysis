"""
================================================================================
CA-Markov Scenario Modeling (2024-2050)
================================================================================

Objective C: Future Habitat Loss Projections - Greater Kafue Ecosystem

This script implements Cellular Automata-Markov (CA-Markov) modeling for
multi-scenario land cover projections:
    - Business-as-Usual (BAU): Historical transition rates continue
    - Enhanced Conservation: Strengthened protection measures
    - Accelerated Development: Increased agricultural expansion

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - rasterio
    - numpy
    - pandas
    - matplotlib
    - scipy
    - joblib
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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy import ndimage
from scipy.linalg import expm
import joblib
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for CA-Markov scenario modeling."""
    
    # Directories
    BASE_DIR = r"D:/Publication/Habitat Loss in the Greater Kafue Ecosystem"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SUSCEPTIBILITY_DIR = os.path.join(BASE_DIR, "outputs/susceptibility")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/projections")
    FIGURES_DIR = os.path.join(BASE_DIR, "outputs/figures")
    TABLES_DIR = os.path.join(BASE_DIR, "outputs/tables")
    
    # Input files
    LC_2024 = os.path.join(DATA_DIR, "GKE_2024.tif")
    SUSCEPTIBILITY = os.path.join(SUSCEPTIBILITY_DIR, "susceptibility_probability.tif")
    PROTECTION_MASK = os.path.join(DATA_DIR, "protection_status.tif")
    KNP_BOUNDARY = os.path.join(DATA_DIR, "nationalParks.shp")
    
    # Land cover classes
    CLASS_LABELS = {
        1: 'Built-up', 2: 'Forest', 3: 'Cropland',
        4: 'Grassland', 5: 'Bareland', 6: 'Water'
    }
    
    CLASS_COLORS = {
        1: '#E31A1C', 2: '#33A02C', 3: '#FF7F00',
        4: '#B2DF8A', 5: '#CAB2D6', 6: '#1F78B4'
    }
    
    N_CLASSES = 6
    NATURAL_CLASSES = [2, 4, 6]
    DISTURBED_CLASSES = [1, 3, 5]
    
    # Projection parameters
    BASE_YEAR = 2024
    TARGET_YEARS = [2030, 2040, 2050]
    
    # CA-Markov parameters
    NEIGHBORHOOD_SIZE = 5
    CONTIGUITY_WEIGHT = 0.5
    N_ENSEMBLE = 10
    RANDOM_STATE = 42
    
    # Historical transition matrix (1984-2024, 40 years)
    # Row i, Column j = P(class i → class j)
    TRANSITION_MATRIX_40YR = np.array([
        [0.6234, 0.0512, 0.2156, 0.0823, 0.0234, 0.0041],  # Built-up
        [0.0089, 0.8534, 0.0512, 0.0745, 0.0098, 0.0022],  # Forest
        [0.0423, 0.0234, 0.8912, 0.0312, 0.0098, 0.0021],  # Cropland
        [0.0156, 0.0623, 0.0845, 0.8123, 0.0198, 0.0055],  # Grassland
        [0.0234, 0.0156, 0.1234, 0.0423, 0.7834, 0.0119],  # Bareland
        [0.0012, 0.0034, 0.0023, 0.0045, 0.0012, 0.9874],  # Water
    ])
    
    # Scenario modifiers
    SCENARIOS = {
        'bau': {
            'name': 'Business-as-Usual',
            'description': 'Historical transition rates continue',
            'modifier': 1.0,
            'protection_factor': 1.0
        },
        'conservation': {
            'name': 'Enhanced Conservation',
            'description': 'Strengthened protection, reduced conversion',
            'modifier': 0.5,
            'protection_factor': 0.3
        },
        'development': {
            'name': 'Accelerated Development',
            'description': 'Increased agricultural expansion',
            'modifier': 1.5,
            'protection_factor': 1.2
        }
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_output_directories():
    """Create output directories."""
    dirs = [Config.OUTPUT_DIR, Config.FIGURES_DIR, Config.TABLES_DIR]
    for scenario in Config.SCENARIOS.keys():
        dirs.append(os.path.join(Config.OUTPUT_DIR, scenario))
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_raster(filepath):
    """Load raster data."""
    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def save_raster(data, profile, filepath):
    """Save raster data."""
    profile.update(dtype=data.dtype, count=1)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data, 1)


def annualize_matrix(trans_matrix, n_years):
    """Convert multi-year transition matrix to annual rates."""
    from scipy.linalg import logm
    
    try:
        P_adj = trans_matrix + 1e-10
        P_adj = P_adj / P_adj.sum(axis=1, keepdims=True)
        
        log_P = logm(P_adj)
        annual_log = log_P / n_years
        annual = expm(annual_log)
        
        annual = np.real(annual)
        annual = np.clip(annual, 0, 1)
        annual = annual / annual.sum(axis=1, keepdims=True)
        
    except Exception:
        identity = np.eye(trans_matrix.shape[0])
        annual = identity + (trans_matrix - identity) / n_years
        annual = np.clip(annual, 0, 1)
        annual = annual / annual.sum(axis=1, keepdims=True)
    
    return annual


# =============================================================================
# CA-MARKOV SIMULATION
# =============================================================================

def create_neighborhood_kernel(size):
    """Create neighborhood kernel for contiguity calculation."""
    kernel = np.ones((size, size))
    kernel[size//2, size//2] = 0
    return kernel / kernel.sum()


def calculate_neighborhood_composition(lc_map, kernel):
    """Calculate neighborhood class composition for each pixel."""
    n_classes = Config.N_CLASSES
    composition = {}
    
    for c in range(1, n_classes + 1):
        class_mask = (lc_map == c).astype(float)
        composition[c] = ndimage.convolve(class_mask, kernel, mode='constant')
    
    return composition


def ca_markov_step(current_lc, trans_matrix, susceptibility, protection_mask,
                   neighborhood_composition, w_contiguity, random_state=None):
    """
    Perform one annual step of CA-Markov simulation.
    
    Args:
        current_lc: Current land cover array
        trans_matrix: Annual transition matrix
        susceptibility: Susceptibility probability map
        protection_mask: Protected area mask
        neighborhood_composition: Pre-calculated neighborhood compositions
        w_contiguity: Weight for neighborhood effect
        random_state: Random seed
        
    Returns:
        ndarray: Updated land cover
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    new_lc = current_lc.copy()
    n_classes = Config.N_CLASSES
    
    valid_mask = (current_lc > 0) & (current_lc <= n_classes)
    valid_idx = np.where(valid_mask)
    
    # Shuffle processing order
    indices = list(range(len(valid_idx[0])))
    np.random.shuffle(indices)
    
    for idx in indices:
        row, col = valid_idx[0][idx], valid_idx[1][idx]
        current_class = current_lc[row, col]
        
        # Skip protected areas (reduced probability)
        if protection_mask is not None and protection_mask[row, col]:
            if np.random.random() > 0.1:  # 90% chance to stay unchanged
                continue
        
        # Get base transition probabilities
        base_probs = trans_matrix[current_class - 1].copy()
        
        # Modify by susceptibility
        susc = susceptibility[row, col]
        if not np.isnan(susc):
            # Increase probability of transition to disturbed classes
            for c in Config.DISTURBED_CLASSES:
                base_probs[c-1] *= (1 + susc)
        
        # Modify by neighborhood
        for c in range(1, n_classes + 1):
            neigh_prop = neighborhood_composition[c][row, col]
            base_probs[c-1] = (1 - w_contiguity) * base_probs[c-1] + w_contiguity * neigh_prop
        
        # Normalize
        base_probs = np.clip(base_probs, 0, None)
        if base_probs.sum() > 0:
            base_probs = base_probs / base_probs.sum()
        else:
            continue
        
        # Sample new class
        new_class = np.random.choice(range(1, n_classes + 1), p=base_probs)
        new_lc[row, col] = new_class
    
    return new_lc


def run_scenario(scenario_key, lc_2024, susceptibility, protection_mask, profile):
    """
    Run CA-Markov simulation for a specific scenario.
    
    Args:
        scenario_key: Scenario identifier
        lc_2024: Base year land cover
        susceptibility: Susceptibility map
        protection_mask: Protection mask
        profile: Raster profile
        
    Returns:
        dict: Projection results
    """
    scenario = Config.SCENARIOS[scenario_key]
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"{'='*60}")
    print(f"  {scenario['description']}")
    
    # Modify transition matrix for scenario
    base_annual = annualize_matrix(Config.TRANSITION_MATRIX_40YR, 40)
    
    # Apply scenario modifier to off-diagonal elements
    modified_annual = base_annual.copy()
    for i in range(Config.N_CLASSES):
        for j in range(Config.N_CLASSES):
            if i != j:
                # Natural → Disturbed transitions
                if (i+1) in Config.NATURAL_CLASSES and (j+1) in Config.DISTURBED_CLASSES:
                    modified_annual[i, j] *= scenario['modifier']
    
    # Re-normalize
    modified_annual = np.clip(modified_annual, 0, 1)
    modified_annual = modified_annual / modified_annual.sum(axis=1, keepdims=True)
    
    # Create neighborhood kernel
    kernel = create_neighborhood_kernel(Config.NEIGHBORHOOD_SIZE)
    
    # Run ensemble simulations
    results = {year: [] for year in Config.TARGET_YEARS}
    
    for run in range(Config.N_ENSEMBLE):
        print(f"\n  Ensemble run {run+1}/{Config.N_ENSEMBLE}")
        
        current_lc = lc_2024.copy()
        current_year = Config.BASE_YEAR
        
        for target_year in Config.TARGET_YEARS:
            n_steps = target_year - current_year
            
            for step in range(n_steps):
                # Update neighborhood composition
                composition = calculate_neighborhood_composition(current_lc, kernel)
                
                # Run simulation step
                current_lc = ca_markov_step(
                    current_lc, modified_annual, susceptibility,
                    protection_mask, composition, Config.CONTIGUITY_WEIGHT,
                    random_state=Config.RANDOM_STATE + run * 1000 + step
                )
            
            results[target_year].append(current_lc.copy())
            current_year = target_year
        
        print(f"    Completed: {Config.TARGET_YEARS}")
    
    # Calculate ensemble statistics
    ensemble_results = {}
    for year in Config.TARGET_YEARS:
        stack = np.stack(results[year], axis=0)
        
        # Modal class (most frequent)
        modal = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), minlength=Config.N_CLASSES+1)[1:].argmax() + 1,
            axis=0, arr=stack
        )
        
        ensemble_results[year] = {
            'modal': modal,
            'ensemble': results[year]
        }
        
        # Save modal projection
        output_path = os.path.join(
            Config.OUTPUT_DIR, scenario_key, f'projection_{year}.tif'
        )
        save_raster(modal.astype(np.uint8), profile, output_path)
        print(f"  ✓ Saved: projection_{year}.tif")
    
    return ensemble_results


def calculate_projection_statistics(lc_2024, projections, profile):
    """Calculate area statistics for projections."""
    pixel_area_ha = abs(profile['transform'][0] * profile['transform'][4]) / 10000
    
    all_stats = []
    
    for scenario_key, scenario_results in projections.items():
        scenario_name = Config.SCENARIOS[scenario_key]['name']
        
        for year, result in scenario_results.items():
            modal = result['modal']
            
            for class_id, class_name in Config.CLASS_LABELS.items():
                count = np.sum(modal == class_id)
                area_ha = count * pixel_area_ha
                
                # Calculate change from 2024
                count_2024 = np.sum(lc_2024 == class_id)
                area_2024 = count_2024 * pixel_area_ha
                change_ha = area_ha - area_2024
                change_pct = (change_ha / area_2024 * 100) if area_2024 > 0 else 0
                
                all_stats.append({
                    'Scenario': scenario_name,
                    'Year': year,
                    'Class': class_name,
                    'Area_ha': area_ha,
                    'Area_2024_ha': area_2024,
                    'Change_ha': change_ha,
                    'Change_pct': change_pct
                })
    
    return pd.DataFrame(all_stats)


def create_comparison_figure(lc_2024, projections, profile):
    """Create multi-panel comparison figure."""
    print("\n--- Creating Comparison Figure ---")
    
    n_scenarios = len(Config.SCENARIOS)
    n_years = len(Config.TARGET_YEARS) + 1
    
    fig, axes = plt.subplots(n_scenarios, n_years, figsize=(4*n_years, 4*n_scenarios))
    
    # Create colormap
    colors = [Config.CLASS_COLORS[i] for i in range(1, Config.N_CLASSES + 1)]
    cmap = ListedColormap(colors)
    bounds = np.arange(0.5, Config.N_CLASSES + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    for i, (scenario_key, scenario_results) in enumerate(projections.items()):
        scenario_name = Config.SCENARIOS[scenario_key]['name']
        
        # 2024 baseline
        ax = axes[i, 0]
        ax.imshow(lc_2024, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f'2024 (Baseline)' if i == 0 else '')
        ax.set_ylabel(scenario_name, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Projections
        for j, year in enumerate(Config.TARGET_YEARS):
            ax = axes[i, j+1]
            modal = scenario_results[year]['modal']
            ax.imshow(modal, cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_title(str(year) if i == 0 else '')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Legend
    legend_elements = [Patch(facecolor=Config.CLASS_COLORS[i], label=name)
                       for i, name in Config.CLASS_LABELS.items()]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.12, 0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'Figure_Scenario_Comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Comparison figure saved")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("CA-MARKOV SCENARIO MODELING")
    print("Greater Kafue Ecosystem (2024-2050)")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    create_output_directories()
    
    # Load inputs
    print("\n--- Loading Input Data ---")
    lc_2024, profile = load_raster(Config.LC_2024)
    print(f"  Land cover 2024: {lc_2024.shape}")
    
    susceptibility, _ = load_raster(Config.SUSCEPTIBILITY)
    print(f"  Susceptibility: {susceptibility.shape}")
    
    # Load or create protection mask
    if os.path.exists(Config.PROTECTION_MASK):
        protection_mask, _ = load_raster(Config.PROTECTION_MASK)
        protection_mask = protection_mask > 0
    else:
        protection_mask = np.zeros_like(lc_2024, dtype=bool)
    print(f"  Protected pixels: {np.sum(protection_mask):,}")
    
    # Run scenarios
    projections = {}
    for scenario_key in Config.SCENARIOS.keys():
        projections[scenario_key] = run_scenario(
            scenario_key, lc_2024, susceptibility, protection_mask, profile
        )
    
    # Statistics
    print("\n" + "="*60)
    print("CALCULATING STATISTICS")
    print("="*60)
    
    stats_df = calculate_projection_statistics(lc_2024, projections, profile)
    stats_df.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Projection_Statistics.csv'), index=False)
    print("  ✓ Statistics saved")
    
    # Natural habitat summary
    natural_summary = stats_df[stats_df['Class'].isin(['Forest', 'Grassland', 'Water'])]
    natural_by_scenario = natural_summary.groupby(['Scenario', 'Year'])['Area_ha'].sum().reset_index()
    natural_by_scenario.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Natural_Habitat_Projections.csv'), index=False)
    
    # Visualization
    create_comparison_figure(lc_2024, projections, profile)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  Projections: {Config.OUTPUT_DIR}")
    print(f"  Tables: {Config.TABLES_DIR}")
    print(f"  Figures: {Config.FIGURES_DIR}")
    
    return projections, stats_df


if __name__ == "__main__":
    projections, stats = main()
