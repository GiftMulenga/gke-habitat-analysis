"""
================================================================================
KNP Boundary Gradient Analysis Visualization
================================================================================

Publication-quality visualizations for edge effect analysis along the
Kafue National Park boundary (1984-2024).

This script generates:
    - Natural habitat gradient profiles
    - Temporal gradient strength analysis
    - Multi-class gradient comparisons
    - Gradient strength rankings
    - Distance zone composition analysis

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scipy

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for KNP gradient visualization."""
    
    # Paths - UPDATE THESE
    INPUT_DIR = Path('D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/tables')
    OUTPUT_DIR = Path('D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/figures')
    
    # KNP gradient files
    KNP_FILES = {
        1984: 'Table_KNP_Gradient_1984_All_Classes.csv',
        1994: 'Table_KNP_Gradient_1994_All_Classes.csv',
        2004: 'Table_KNP_Gradient_2004_All_Classes.csv',
        2014: 'Table_KNP_Gradient_2014_All_Classes.csv',
        2024: 'Table_KNP_Gradient_2024_All_Classes.csv'
    }
    
    # Land cover classes
    NATURAL_CLASSES = ['Forest_%', 'Grassland_%', 'Water_%']
    DISTURBED_CLASSES = ['Built-up_%', 'Cropland_%']
    ALL_CLASSES = ['Built-up_%', 'Forest_%', 'Cropland_%', 'Grassland_%', 'Bareland_%', 'Water_%']
    
    # Color scheme
    COLORS = {
        'Built-up_%': '#E74C3C',
        'Forest_%': '#27AE60',
        'Cropland_%': '#F39C12',
        'Grassland_%': '#95A5A6',
        'Bareland_%': '#BDC3C7',
        'Water_%': '#3498DB',
        'Natural_Habitat_%': '#229954',
        'Disturbed_%': '#C0392B'
    }
    
    # Year colors
    YEAR_COLORS = {
        1984: '#1f77b4',
        1994: '#ff7f0e',
        2004: '#2ca02c',
        2014: '#d62728',
        2024: '#9467bd'
    }
    
    DPI = 300


# =============================================================================
# DATA LOADING
# =============================================================================

def load_knp_gradient_data():
    """
    Load all KNP gradient data across time periods.
    
    Returns:
        dict: Dictionary with years as keys and DataFrames as values
    """
    print("Loading KNP gradient data...")
    data = {}
    
    for year, filename in Config.KNP_FILES.items():
        filepath = Config.INPUT_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            data[year] = df
            print(f"  Loaded {year}: {len(df)} distance bands")
        else:
            print(f"  Warning: {filename} not found")
    
    return data


# =============================================================================
# GRADIENT ANALYSIS FUNCTIONS
# =============================================================================

def calculate_gradient_metrics(df, class_name, distance_col='Distance_m'):
    """
    Calculate gradient metrics for a specific land cover class.
    
    Args:
        df: DataFrame with gradient data
        class_name: Column name for the class
        distance_col: Column name for distance
        
    Returns:
        dict: Gradient statistics
    """
    valid_data = df[[distance_col, class_name]].dropna()
    
    if len(valid_data) < 2:
        return None
    
    x = valid_data[distance_col].values
    y = valid_data[class_name].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    value_at_0m = y[0] if len(y) > 0 else np.nan
    value_at_max = y[-1] if len(y) > 0 else np.nan
    total_change = value_at_max - value_at_0m
    
    # Calculate early and late zone slopes
    early_zone = valid_data[valid_data[distance_col] <= 5000]
    late_zone = valid_data[(valid_data[distance_col] > 5000) & (valid_data[distance_col] <= 10000)]
    
    early_slope = np.nan
    late_slope = np.nan
    
    if len(early_zone) >= 2:
        early_slope = stats.linregress(early_zone[distance_col], early_zone[class_name])[0]
    
    if len(late_zone) >= 2:
        late_slope = stats.linregress(late_zone[distance_col], late_zone[class_name])[0]
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'value_0m': value_at_0m,
        'value_max': value_at_max,
        'total_change': total_change,
        'early_slope_0_5000m': early_slope,
        'late_slope_5000_10000m': late_slope,
        'gradient_strength': abs(slope)
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_natural_habitat_gradient(data_dict, output_path):
    """
    Create natural habitat gradient plot for all years.
    
    Args:
        data_dict: Dictionary of DataFrames by year
        output_path: Path to save the figure
    """
    print("\n  Creating: Natural Habitat Gradient...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    years = sorted(data_dict.keys())
    year_colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    
    for i, year in enumerate(years):
        df = data_dict[year]
        ax.plot(df['Distance_m'], df['Natural_Habitat_%'],
                marker='o', markersize=3, linewidth=2.5,
                label=str(year), color=year_colors[i], alpha=0.8)
    
    # Gradient zones
    ax.axvspan(0, 5000, alpha=0.1, color='red', label='Steep gradient (0-5 km)')
    ax.axvspan(5000, 10000, alpha=0.1, color='orange', label='Moderate (5-10 km)')
    ax.axvspan(10000, 15000, alpha=0.1, color='green', label='Stable (>10 km)')
    
    ax.set_xlabel('Distance from KNP Boundary (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Natural Habitat Cover (%)', fontsize=13, fontweight='bold')
    ax.set_title('Natural Habitat Decline with Distance from KNP Boundary (1984-2024)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.95, fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 15000)
    ax.set_ylim(70, 100)
    
    # Secondary x-axis in km
    ax2 = ax.twiny()
    ax2.set_xlim(0, 15)
    ax2.set_xlabel('Distance from KNP Boundary (km)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def create_gradient_strength_temporal(data_dict, output_path):
    """
    Create gradient strength over time visualization.
    
    Args:
        data_dict: Dictionary of DataFrames by year
        output_path: Path to save the figure
    """
    print("\n  Creating: Gradient Strength Over Time...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    years = sorted(data_dict.keys())
    gradient_metrics = []
    
    for year in years:
        df = data_dict[year]
        metrics = calculate_gradient_metrics(df, 'Natural_Habitat_%')
        if metrics:
            gradient_metrics.append({
                'Year': year,
                'Slope (pp/1000m)': metrics['slope'] * 1000,
                'R2': metrics['r_squared'],
                'Total Decline (pp)': -metrics['total_change']
            })
    
    metrics_df = pd.DataFrame(gradient_metrics)
    
    ax2 = ax.twinx()
    
    bars = ax.bar(metrics_df['Year'], metrics_df['Total Decline (pp)'],
                   width=2, alpha=0.6, color='coral',
                   label='Total decline (0-15 km)', edgecolor='black', linewidth=1.5)
    
    line = ax2.plot(metrics_df['Year'], -metrics_df['Slope (pp/1000m)'],
                    marker='D', markersize=10, linewidth=3,
                    color='darkgreen', label='Gradient strength',
                    markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Natural Habitat Decline (pp)', fontsize=13, fontweight='bold', color='coral')
    ax2.set_ylabel('Gradient Strength (pp/1000m)', fontsize=13, fontweight='bold', color='darkgreen')
    ax.set_title('Edge Effect Intensity Over Time (1984-2024)',
                 fontsize=14, fontweight='bold', pad=20)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95)
    
    ax.tick_params(axis='y', labelcolor='coral')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_xticks(years)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")
    
    return metrics_df


def create_all_classes_gradient(df, year, output_path):
    """
    Create gradient plot for all classes in a single year.
    
    Args:
        df: DataFrame for the year
        year: Year value
        output_path: Path to save the figure
    """
    print(f"\n  Creating: All Classes Gradient ({year})...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for class_name in Config.ALL_CLASSES:
        if class_name in df.columns:
            color = Config.COLORS.get(class_name, 'gray')
            label = class_name.replace('_%', '')
            ax.plot(df['Distance_m'], df[class_name],
                   marker='o', markersize=3, linewidth=2,
                   label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Distance from KNP Boundary (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Land Cover (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Land Cover Composition by Distance from KNP ({year})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 15000)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def create_natural_vs_disturbed(data_dict, output_path):
    """
    Create Natural vs Disturbed comparison across all years.
    
    Args:
        data_dict: Dictionary of DataFrames by year
        output_path: Path to save the figure
    """
    print("\n  Creating: Natural vs Disturbed Comparison...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    years = sorted(data_dict.keys())
    
    for year in years:
        df = data_dict[year]
        color = Config.YEAR_COLORS.get(year, 'gray')
        
        ax1.plot(df['Distance_m'], df['Natural_Habitat_%'],
                marker='o', markersize=4, linewidth=2.5,
                label=str(year), color=color, alpha=0.8)
        
        ax2.plot(df['Distance_m'], df['Disturbed_%'],
                marker='s', markersize=4, linewidth=2.5,
                label=str(year), color=color, alpha=0.8)
    
    ax1.set_xlabel('Distance from KNP (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Natural Habitat (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Natural Habitat', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 15000)
    ax1.set_ylim(70, 100)
    
    ax2.set_xlabel('Distance from KNP (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Disturbed (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Disturbed Areas', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 15000)
    
    plt.suptitle('Natural Habitat vs Disturbed Areas: Temporal Comparison (1984-2024)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def generate_summary_statistics(data_dict, output_path):
    """
    Generate summary statistics table.
    
    Args:
        data_dict: Dictionary of DataFrames by year
        output_path: Path to save the CSV
        
    Returns:
        DataFrame: Summary statistics
    """
    print("\n  Generating summary statistics...")
    
    results = []
    
    for year in sorted(data_dict.keys()):
        df = data_dict[year]
        
        for class_name in ['Natural_Habitat_%', 'Forest_%', 'Cropland_%', 
                           'Disturbed_%', 'Built-up_%', 'Grassland_%']:
            if class_name in df.columns:
                metrics = calculate_gradient_metrics(df, class_name)
                if metrics:
                    results.append({
                        'Year': year,
                        'Class': class_name.replace('_%', ''),
                        'Value_0m': metrics['value_0m'],
                        'Value_15km': metrics['value_max'],
                        'Total_Change_pp': metrics['total_change'],
                        'Slope_pp_km': metrics['slope'] * 1000,
                        'R_squared': metrics['r_squared'],
                        'Early_Slope_0_5km': metrics['early_slope_0_5000m'] * 1000 if metrics['early_slope_0_5000m'] else np.nan,
                        'Late_Slope_5_10km': metrics['late_slope_5000_10000m'] * 1000 if metrics['late_slope_5000_10000m'] else np.nan
                    })
    
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path}")
    
    return summary_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("KNP BOUNDARY GRADIENT ANALYSIS VISUALIZATION")
    print("="*80)
    
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        # Load data
        knp_data = load_knp_gradient_data()
        
        if not knp_data:
            print("\nError: No data files found.")
            return
        
        # Generate visualizations
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Natural habitat gradient
        create_natural_habitat_gradient(
            knp_data,
            Config.OUTPUT_DIR / 'Figure_Natural_Habitat_Gradient.png'
        )
        
        # Gradient strength over time
        metrics_df = create_gradient_strength_temporal(
            knp_data,
            Config.OUTPUT_DIR / 'Figure_Gradient_Strength_Temporal.png'
        )
        
        # Save metrics
        metrics_df.to_csv(Config.OUTPUT_DIR / 'Table_Gradient_Metrics_Temporal.csv', index=False)
        
        # All classes for each year
        for year in sorted(knp_data.keys()):
            create_all_classes_gradient(
                knp_data[year],
                year,
                Config.OUTPUT_DIR / f'Figure_All_Classes_{year}.png'
            )
        
        # Natural vs Disturbed
        create_natural_vs_disturbed(
            knp_data,
            Config.OUTPUT_DIR / 'Figure_Natural_vs_Disturbed_All_Years.png'
        )
        
        # Summary statistics
        generate_summary_statistics(
            knp_data,
            Config.OUTPUT_DIR / 'Table_Gradient_Summary_Statistics.csv'
        )
        
        # Key findings
        print("\n" + "="*80)
        print("KEY FINDINGS (2024)")
        print("="*80)
        
        if 2024 in knp_data:
            df_2024 = knp_data[2024]
            nat_metrics = calculate_gradient_metrics(df_2024, 'Natural_Habitat_%')
            
            if nat_metrics:
                print(f"\nNatural Habitat Gradient:")
                print(f"  Value at 0m: {nat_metrics['value_0m']:.1f}%")
                print(f"  Value at 15km: {nat_metrics['value_max']:.1f}%")
                print(f"  Total decline: {-nat_metrics['total_change']:.1f} pp")
                print(f"  Gradient slope: {nat_metrics['slope']*1000:.3f} pp/km")
                print(f"  R-squared: {nat_metrics['r_squared']:.3f}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"Outputs saved to: {Config.OUTPUT_DIR}")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
