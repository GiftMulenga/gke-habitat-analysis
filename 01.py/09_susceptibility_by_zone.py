"""
================================================================================
Susceptibility Analysis by Management Zone
================================================================================

Analyzes habitat loss susceptibility patterns across different management
zones within the Greater Kafue Ecosystem.

Management Zones:
    - Kafue National Park (KNP) - Core protected area
    - Game Management Areas (GMAs) - Buffer zones
    - Open Areas - Unprotected lands

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - rasterio
    - numpy
    - pandas
    - geopandas
    - matplotlib
    - seaborn

================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import mapping
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for zone-based susceptibility analysis."""
    
    # Directories
    BASE_DIR = r"D:/Publication/Habitat Loss in the Greater Kafue Ecosystem"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SUSCEPTIBILITY_DIR = os.path.join(BASE_DIR, "outputs/susceptibility")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/zones")
    FIGURES_DIR = os.path.join(BASE_DIR, "outputs/figures")
    TABLES_DIR = os.path.join(BASE_DIR, "outputs/tables")
    
    # Input files
    SUSCEPTIBILITY_RASTER = os.path.join(SUSCEPTIBILITY_DIR, "susceptibility_probability.tif")
    GKE_BOUNDARY = os.path.join(DATA_DIR, "GKE_Boundary.shp")
    KNP_BOUNDARY = os.path.join(DATA_DIR, "nationalParks.shp")
    GMA_BOUNDARY = os.path.join(DATA_DIR, "GMA.shp")
    
    # Susceptibility classes
    SUSCEPTIBILITY_BREAKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    SUSCEPTIBILITY_LABELS = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    
    # Zone colors
    ZONE_COLORS = {
        'KNP': '#2E7D32',
        'GMA': '#FFA726',
        'Open Area': '#EF5350'
    }
    
    # Visualization
    FIGURE_DPI = 300


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_output_directories():
    """Create output directories."""
    for d in [Config.OUTPUT_DIR, Config.FIGURES_DIR, Config.TABLES_DIR]:
        os.makedirs(d, exist_ok=True)


def load_susceptibility():
    """Load susceptibility raster."""
    with rasterio.open(Config.SUSCEPTIBILITY_RASTER) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return data, profile, transform, crs


def load_zone_boundaries():
    """Load and prepare zone boundary files."""
    zones = {}
    
    # GKE boundary
    if os.path.exists(Config.GKE_BOUNDARY):
        zones['gke'] = gpd.read_file(Config.GKE_BOUNDARY)
    
    # KNP boundary
    if os.path.exists(Config.KNP_BOUNDARY):
        knp = gpd.read_file(Config.KNP_BOUNDARY)
        if 'NAME' in knp.columns:
            knp = knp[knp['NAME'].str.contains('Kafue', case=False, na=False)]
        zones['knp'] = knp
    
    # GMA boundaries
    if os.path.exists(Config.GMA_BOUNDARY):
        zones['gma'] = gpd.read_file(Config.GMA_BOUNDARY)
    
    return zones


# =============================================================================
# ZONE ANALYSIS
# =============================================================================

def create_zone_mask(susceptibility, zones, transform, crs):
    """
    Create zone mask raster.
    
    Zone codes:
        1 = KNP
        2 = GMA
        3 = Open Area
        0 = Outside study area
    
    Args:
        susceptibility: Susceptibility array
        zones: Zone GeoDataFrames
        transform: Raster transform
        crs: Coordinate reference system
        
    Returns:
        ndarray: Zone mask
    """
    shape = susceptibility.shape
    zone_mask = np.zeros(shape, dtype=np.uint8)
    
    # Start with entire GKE as Open Area (3)
    if 'gke' in zones:
        gke_geom = zones['gke'].to_crs(crs).geometry.values
        gke_raster = rasterize(
            [(geom, 3) for geom in gke_geom],
            out_shape=shape, transform=transform, fill=0
        )
        zone_mask = np.where(gke_raster > 0, gke_raster, zone_mask)
    
    # Overlay GMAs (2)
    if 'gma' in zones:
        gma_geom = zones['gma'].to_crs(crs).geometry.values
        gma_raster = rasterize(
            [(geom, 2) for geom in gma_geom],
            out_shape=shape, transform=transform, fill=0
        )
        zone_mask = np.where(gma_raster > 0, gma_raster, zone_mask)
    
    # Overlay KNP (1) - highest priority
    if 'knp' in zones:
        knp_geom = zones['knp'].to_crs(crs).geometry.values
        knp_raster = rasterize(
            [(geom, 1) for geom in knp_geom],
            out_shape=shape, transform=transform, fill=0
        )
        zone_mask = np.where(knp_raster > 0, knp_raster, zone_mask)
    
    return zone_mask


def calculate_zone_statistics(susceptibility, zone_mask, pixel_area_ha):
    """
    Calculate susceptibility statistics by zone.
    
    Args:
        susceptibility: Susceptibility probability array
        zone_mask: Zone mask array
        pixel_area_ha: Pixel area in hectares
        
    Returns:
        DataFrame: Zone statistics
    """
    zone_names = {1: 'KNP', 2: 'GMA', 3: 'Open Area'}
    
    results = []
    
    for zone_code, zone_name in zone_names.items():
        zone_pixels = zone_mask == zone_code
        zone_susc = susceptibility[zone_pixels & ~np.isnan(susceptibility)]
        
        if len(zone_susc) == 0:
            continue
        
        # Basic statistics
        stats = {
            'Zone': zone_name,
            'N_Pixels': len(zone_susc),
            'Area_ha': len(zone_susc) * pixel_area_ha,
            'Mean_Susceptibility': np.mean(zone_susc),
            'Median_Susceptibility': np.median(zone_susc),
            'Std_Susceptibility': np.std(zone_susc),
            'Min_Susceptibility': np.min(zone_susc),
            'Max_Susceptibility': np.max(zone_susc)
        }
        
        # Susceptibility class distribution
        for i, label in enumerate(Config.SUSCEPTIBILITY_LABELS):
            low = Config.SUSCEPTIBILITY_BREAKS[i]
            high = Config.SUSCEPTIBILITY_BREAKS[i + 1]
            count = np.sum((zone_susc >= low) & (zone_susc < high))
            stats[f'Pct_{label.replace(" ", "_")}'] = count / len(zone_susc) * 100
        
        results.append(stats)
    
    return pd.DataFrame(results)


def calculate_susceptibility_by_gma(susceptibility, zones, transform, crs, pixel_area_ha):
    """
    Calculate susceptibility for each individual GMA.
    
    Args:
        susceptibility: Susceptibility array
        zones: Zone GeoDataFrames
        transform: Raster transform
        crs: Coordinate reference system
        pixel_area_ha: Pixel area in hectares
        
    Returns:
        DataFrame: GMA-level statistics
    """
    if 'gma' not in zones:
        return pd.DataFrame()
    
    gma_gdf = zones['gma'].to_crs(crs)
    results = []
    
    for idx, row in gma_gdf.iterrows():
        gma_name = row.get('NAME', row.get('name', f'GMA_{idx}'))
        
        # Rasterize individual GMA
        gma_mask = rasterize(
            [(row.geometry, 1)],
            out_shape=susceptibility.shape,
            transform=transform, fill=0
        ) > 0
        
        gma_susc = susceptibility[gma_mask & ~np.isnan(susceptibility)]
        
        if len(gma_susc) == 0:
            continue
        
        stats = {
            'GMA': gma_name,
            'Area_ha': len(gma_susc) * pixel_area_ha,
            'Mean_Susceptibility': np.mean(gma_susc),
            'Pct_High_VeryHigh': np.sum(gma_susc >= 0.6) / len(gma_susc) * 100
        }
        
        results.append(stats)
    
    df = pd.DataFrame(results)
    return df.sort_values('Mean_Susceptibility', ascending=False)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_zone_comparison(stats_df):
    """Create zone comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean susceptibility by zone
    ax = axes[0]
    zones = stats_df['Zone'].values
    means = stats_df['Mean_Susceptibility'].values
    stds = stats_df['Std_Susceptibility'].values
    colors = [Config.ZONE_COLORS.get(z, 'gray') for z in zones]
    
    bars = ax.bar(zones, means, yerr=stds, color=colors, capsize=5, edgecolor='black')
    ax.set_ylabel('Mean Susceptibility', fontsize=12)
    ax.set_title('Susceptibility by Management Zone', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax.legend()
    
    # Class distribution by zone
    ax = axes[1]
    class_cols = [c for c in stats_df.columns if c.startswith('Pct_')]
    class_labels = [c.replace('Pct_', '').replace('_', ' ') for c in class_cols]
    
    x = np.arange(len(zones))
    width = 0.15
    
    for i, (col, label) in enumerate(zip(class_cols, class_labels)):
        offset = (i - len(class_cols)/2 + 0.5) * width
        ax.bar(x + offset, stats_df[col], width, label=label)
    
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Susceptibility Class Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'Figure_Zone_Susceptibility.png'),
                dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_gma_ranking(gma_df):
    """Create GMA ranking visualization."""
    if gma_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by susceptibility
    gma_df = gma_df.sort_values('Mean_Susceptibility', ascending=True)
    
    colors = ['#EF5350' if s >= 0.6 else '#FFA726' if s >= 0.4 else '#66BB6A' 
              for s in gma_df['Mean_Susceptibility']]
    
    ax.barh(gma_df['GMA'], gma_df['Mean_Susceptibility'], color=colors, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Mean Susceptibility', fontsize=12)
    ax.set_title('GMA Susceptibility Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURES_DIR, 'Figure_GMA_Ranking.png'),
                dpi=Config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("SUSCEPTIBILITY ANALYSIS BY MANAGEMENT ZONE")
    print("Greater Kafue Ecosystem")
    print("="*60)
    
    create_output_directories()
    
    # Load data
    print("\n--- Loading Data ---")
    susceptibility, profile, transform, crs = load_susceptibility()
    print(f"  Susceptibility shape: {susceptibility.shape}")
    
    zones = load_zone_boundaries()
    print(f"  Zones loaded: {list(zones.keys())}")
    
    pixel_area_ha = abs(profile['transform'][0] * profile['transform'][4]) / 10000
    
    # Create zone mask
    print("\n--- Creating Zone Mask ---")
    zone_mask = create_zone_mask(susceptibility, zones, transform, crs)
    print(f"  KNP pixels: {np.sum(zone_mask == 1):,}")
    print(f"  GMA pixels: {np.sum(zone_mask == 2):,}")
    print(f"  Open Area pixels: {np.sum(zone_mask == 3):,}")
    
    # Calculate statistics
    print("\n--- Calculating Zone Statistics ---")
    stats_df = calculate_zone_statistics(susceptibility, zone_mask, pixel_area_ha)
    
    for _, row in stats_df.iterrows():
        print(f"\n  {row['Zone']}:")
        print(f"    Area: {row['Area_ha']:,.0f} ha")
        print(f"    Mean susceptibility: {row['Mean_Susceptibility']:.3f}")
        print(f"    High/Very High: {row.get('Pct_High', 0) + row.get('Pct_Very_High', 0):.1f}%")
    
    stats_df.to_csv(os.path.join(Config.TABLES_DIR, 'Table_Zone_Susceptibility.csv'), index=False)
    
    # GMA-level analysis
    print("\n--- Calculating GMA Statistics ---")
    gma_df = calculate_susceptibility_by_gma(susceptibility, zones, transform, crs, pixel_area_ha)
    
    if not gma_df.empty:
        gma_df.to_csv(os.path.join(Config.TABLES_DIR, 'Table_GMA_Susceptibility.csv'), index=False)
        print(f"  Analyzed {len(gma_df)} GMAs")
    
    # Visualizations
    print("\n--- Creating Visualizations ---")
    plot_zone_comparison(stats_df)
    print("  ✓ Zone comparison figure saved")
    
    plot_gma_ranking(gma_df)
    print("  ✓ GMA ranking figure saved")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return {'stats': stats_df, 'gma': gma_df}


if __name__ == "__main__":
    results = main()
