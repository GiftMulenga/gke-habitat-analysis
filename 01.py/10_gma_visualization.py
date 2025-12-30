"""
================================================================================
GMA Land Cover Change Visualization Suite
================================================================================

Publication-quality visualizations for Game Management Area (GMA) land cover
change analysis in the Greater Kafue Ecosystem (1984-2024).

This script generates:
    - Natural vs Disturbed habitat trend analysis
    - 1984 vs 2024 composition comparison
    - Area change heatmaps
    - Forest cover trajectory analysis
    - Habitat conversion waterfall charts
    - Forest-Cropland dual-axis trends

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - pandas
    - numpy
    - matplotlib
    - seaborn

================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for GMA visualization."""
    
    # Paths - UPDATE THESE
    INPUT_FILE = "D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/tables/Table_All_Individual_GMAs_LULC.csv"
    OUTPUT_DIR = "D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/figures"
    
    # Figure settings
    DPI = 300
    FIGSIZE_LARGE = (16, 10)
    FIGSIZE_MEDIUM = (14, 8)
    FIGSIZE_SMALL = (12, 8)
    
    # Color scheme for land cover classes
    COLORS = {
        'Built-up': '#8B0000',
        'Forest': '#228B22',
        'Cropland': '#FFD700',
        'Grassland': '#90EE90',
        'Bareland': '#D2691E',
        'Water': '#4169E1'
    }
    
    # Land cover class order
    LAND_COVER_CLASSES = ['Built-up', 'Forest', 'Cropland', 'Grassland', 'Bareland', 'Water']
    
    # Time periods
    YEARS = [1984, 1994, 2004, 2014, 2024]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data(filepath):
    """
    Load CSV and prepare data for analysis.
    
    Args:
        filepath: Path to input CSV file
        
    Returns:
        tuple: (full DataFrame, clean DataFrame, GMA list)
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Get only land cover class rows
    df_clean = df[df['Land Cover Class'].isin(Config.LAND_COVER_CLASSES)].copy()
    
    # Get unique GMAs
    gmas = df_clean['GMA'].unique()
    print(f"Found {len(gmas)} GMAs: {', '.join(gmas)}")
    
    return df, df_clean, gmas


def create_natural_vs_disturbed_data(df):
    """
    Calculate natural vs disturbed habitat over time.
    
    Args:
        df: Full DataFrame with summary rows
        
    Returns:
        DataFrame: Reshaped data for plotting
    """
    unique_classes = df['Land Cover Class'].unique()
    
    # Find correct labels
    natural_label = None
    disturbed_label = None
    
    for lc in unique_classes:
        if 'natural' in lc.lower() and 'habitat' in lc.lower():
            natural_label = lc
        if 'disturbed' in lc.lower():
            disturbed_label = lc
    
    if natural_label is None or disturbed_label is None:
        return pd.DataFrame(columns=['GMA', 'Year', 'Type', 'Area'])
    
    nat_dist = df[df['Land Cover Class'].isin([natural_label, disturbed_label])].copy()
    
    # Reshape for plotting
    result = []
    gmas = nat_dist['GMA'].unique()
    
    for gma in gmas:
        gma_data = nat_dist[nat_dist['GMA'] == gma]
        for year in Config.YEARS:
            area_col = f'{year} Area (ha)'
            
            if area_col not in gma_data.columns:
                continue
            
            nat_row = gma_data[gma_data['Land Cover Class'] == natural_label]
            dist_row = gma_data[gma_data['Land Cover Class'] == disturbed_label]
            
            if not nat_row.empty:
                result.append({
                    'GMA': gma, 'Year': year,
                    'Type': 'Natural Habitat',
                    'Area': nat_row[area_col].values[0]
                })
            
            if not dist_row.empty:
                result.append({
                    'GMA': gma, 'Year': year,
                    'Type': 'Disturbed',
                    'Area': dist_row[area_col].values[0]
                })
    
    return pd.DataFrame(result)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_natural_vs_disturbed_trends(df, output_dir):
    """
    Create stacked area charts showing Natural vs Disturbed habitat trends.
    
    Args:
        df: Full DataFrame
        output_dir: Output directory path
    """
    print("\nGenerating: Natural vs Disturbed Habitat Trends...")
    
    nat_dist_df = create_natural_vs_disturbed_data(df)
    
    if nat_dist_df.empty:
        print("  No Natural/Disturbed data available. Skipping.")
        return
    
    gmas = nat_dist_df['GMA'].unique()
    n_gmas = len(gmas)
    n_cols = 3
    n_rows = int(np.ceil(n_gmas / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten() if n_gmas > 1 else [axes]
    
    for idx, gma in enumerate(gmas):
        ax = axes[idx]
        gma_data = nat_dist_df[nat_dist_df['GMA'] == gma]
        
        if gma_data.empty:
            ax.set_visible(False)
            continue
        
        try:
            pivot = gma_data.pivot(index='Year', columns='Type', values='Area')
        except Exception:
            ax.set_visible(False)
            continue
        
        if pivot.empty or 'Natural Habitat' not in pivot.columns:
            ax.set_visible(False)
            continue
        
        ax.fill_between(pivot.index, 0, pivot['Natural Habitat'],
                        alpha=0.7, color='#2E7D32', label='Natural Habitat')
        if 'Disturbed' in pivot.columns:
            ax.fill_between(pivot.index, pivot['Natural Habitat'],
                            pivot['Natural Habitat'] + pivot['Disturbed'],
                            alpha=0.7, color='#C62828', label='Disturbed')
        
        ax.set_title(f'{gma} GMA', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Area (ha)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')
    
    for idx in range(n_gmas, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Natural vs Disturbed Habitat Trends (1984-2024) - All GMAs',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_natural_vs_disturbed_trends.png')
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_composition_comparison(df_clean, output_dir):
    """
    Create 1984 vs 2024 composition comparison charts.
    
    Args:
        df_clean: Clean DataFrame with land cover classes only
        output_dir: Output directory path
    """
    print("\nGenerating: 1984 vs 2024 Composition Comparison...")
    
    gmas = df_clean['GMA'].unique()
    n_gmas = len(gmas)
    n_cols = 3
    n_rows = int(np.ceil(n_gmas / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_gmas > 1 else [axes]
    
    for idx, gma in enumerate(gmas):
        ax = axes[idx]
        gma_data = df_clean[df_clean['GMA'] == gma]
        
        data_1984 = []
        data_2024 = []
        
        for lc_class in Config.LAND_COVER_CLASSES:
            class_data = gma_data[gma_data['Land Cover Class'] == lc_class]
            if not class_data.empty:
                data_1984.append(class_data['1984 (%)'].values[0] if '1984 (%)' in class_data.columns else 0)
                data_2024.append(class_data['2024 (%)'].values[0] if '2024 (%)' in class_data.columns else 0)
            else:
                data_1984.append(0)
                data_2024.append(0)
        
        bottom_1984 = np.zeros(1)
        bottom_2024 = np.zeros(1)
        
        for i, lc_class in enumerate(Config.LAND_COVER_CLASSES):
            color = Config.COLORS.get(lc_class, '#808080')
            ax.bar(0, data_1984[i], bottom=bottom_1984, color=color,
                   width=0.6, label=lc_class if idx == 0 else "")
            bottom_1984 += data_1984[i]
            ax.bar(1, data_2024[i], bottom=bottom_2024, color=color, width=0.6)
            bottom_2024 += data_2024[i]
        
        ax.set_title(f'{gma} GMA', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['1984', '2024'], fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', alpha=0.3)
    
    for idx in range(n_gmas, len(axes)):
        axes[idx].set_visible(False)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=6, fontsize=11, frameon=True)
    
    plt.suptitle('Land Cover Composition Change: 1984 vs 2024',
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(output_dir, 'fig_composition_1984_vs_2024.png')
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_area_change_heatmap(df_clean, output_dir):
    """
    Create heatmap showing area change by class and GMA.
    
    Args:
        df_clean: Clean DataFrame
        output_dir: Output directory path
    """
    print("\nGenerating: Area Change Heatmap...")
    
    gmas = df_clean['GMA'].unique()
    
    change_matrix = []
    annotation_matrix = []
    
    for gma in gmas:
        gma_row = []
        annot_row = []
        gma_data = df_clean[df_clean['GMA'] == gma]
        
        for lc_class in Config.LAND_COVER_CLASSES:
            class_data = gma_data[gma_data['Land Cover Class'] == lc_class]
            if not class_data.empty:
                area_1984 = class_data['1984 Area (ha)'].values[0]
                area_2024 = class_data['2024 Area (ha)'].values[0]
                area_change = area_2024 - area_1984
                gma_row.append(area_change)
                
                if abs(area_change) >= 10000:
                    annot_row.append(f'{area_change/1000:.1f}k')
                elif abs(area_change) >= 1000:
                    annot_row.append(f'{area_change/1000:.2f}k')
                else:
                    annot_row.append(f'{area_change:.0f}')
            else:
                gma_row.append(0)
                annot_row.append('0')
        
        change_matrix.append(gma_row)
        annotation_matrix.append(annot_row)
    
    change_df = pd.DataFrame(change_matrix, index=gmas, columns=Config.LAND_COVER_CLASSES)
    annot_df = pd.DataFrame(annotation_matrix, index=gmas, columns=Config.LAND_COVER_CLASSES)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    max_abs = max(abs(change_df.min().min()), abs(change_df.max().max()))
    
    sns.heatmap(change_df, annot=annot_df, fmt='', cmap='RdYlGn',
                center=0, vmin=-max_abs, vmax=max_abs,
                cbar_kws={'label': 'Area Change (ha)\nNegative = Loss, Positive = Gain'},
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    
    ax.set_title('Net Land Cover Area Change by GMA (1984-2024)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Game Management Area', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_heatmap_area_change.png')
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_forest_trajectory(df_clean, output_dir):
    """
    Create slope chart showing forest cover trajectory.
    
    Args:
        df_clean: Clean DataFrame
        output_dir: Output directory path
    """
    print("\nGenerating: Forest Cover Trajectory...")
    
    gmas = df_clean['GMA'].unique()
    forest_data = df_clean[df_clean['Land Cover Class'] == 'Forest'].copy()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors_gma = plt.cm.tab10(np.linspace(0, 1, len(gmas)))
    
    for idx, gma in enumerate(gmas):
        gma_forest = forest_data[forest_data['GMA'] == gma]
        
        if not gma_forest.empty:
            area_1984 = gma_forest['1984 (%)'].values[0]
            area_2024 = gma_forest['2024 (%)'].values[0]
            
            ax.plot([0, 1], [area_1984, area_2024], 'o-',
                   color=colors_gma[idx], linewidth=2, markersize=8,
                   alpha=0.7, label=gma)
            
            ax.text(-0.05, area_1984, f'{area_1984:.1f}%', ha='right', va='center', fontsize=9)
            ax.text(1.05, area_2024, f'{area_2024:.1f}%', ha='left', va='center', fontsize=9)
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['1984', '2024'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Forest Cover (%)', fontsize=12, fontweight='bold')
    ax.set_title('Forest Cover Trajectory by GMA (1984-2024)',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, frameon=True)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_forest_trajectory_slope.png')
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_habitat_waterfall(df, output_dir):
    """
    Create waterfall chart showing natural habitat conversion.
    
    Args:
        df: Full DataFrame
        output_dir: Output directory path
    """
    print("\nGenerating: Habitat Conversion Waterfall...")
    
    nat_habitat = df[df['Land Cover Class'].str.contains('Natural Habitat', case=False, na=False)].copy()
    
    if nat_habitat.empty:
        print("  No Natural Habitat data found. Skipping.")
        return
    
    total_areas = []
    for year in Config.YEARS:
        area_col = f'{year} Area (ha)'
        if area_col in nat_habitat.columns:
            total_areas.append(nat_habitat[area_col].sum())
    
    if len(total_areas) != len(Config.YEARS):
        print("  Incomplete year data. Skipping.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(Config.YEARS))
    
    ax.bar(0, total_areas[0], color='#2E7D32', alpha=0.7, width=0.6)
    ax.text(0, total_areas[0]/2, f'{total_areas[0]:,.0f} ha',
           ha='center', va='center', fontweight='bold', fontsize=10)
    
    for i in range(1, len(Config.YEARS)):
        change = total_areas[i] - total_areas[i-1]
        color = '#C62828' if change < 0 else '#2E7D32'
        
        if change < 0:
            bottom = total_areas[i]
            height = abs(change)
        else:
            bottom = total_areas[i-1]
            height = change
        
        ax.bar(i, height, bottom=bottom, color=color, alpha=0.7, width=0.6)
        ax.plot([i-1, i], [total_areas[i-1], total_areas[i-1]], 'k--', linewidth=1, alpha=0.5)
        
        mid_point = bottom + height/2
        ax.text(i, mid_point, f'{change:,.0f} ha',
               ha='center', va='center', fontweight='bold', fontsize=9)
        ax.text(i, total_areas[i] + 50000, f'{total_areas[i]:,.0f} ha',
               ha='center', va='bottom', fontsize=9, style='italic')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(Config.YEARS, fontsize=11)
    ax.set_ylabel('Natural Habitat Area (ha)', fontsize=12, fontweight='bold')
    ax.set_title('Natural Habitat Conversion - All GMAs Combined (1984-2024)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_waterfall_habitat_conversion.png')
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_forest_cropland_dual_axis(df_clean, output_dir):
    """
    Create dual-axis plots showing Forest vs Cropland trends.
    
    Args:
        df_clean: Clean DataFrame
        output_dir: Output directory path
    """
    print("\nGenerating: Forest vs Cropland Trends...")
    
    key_gmas = ['Bilili', 'Namwala', 'Sichifulo', 'Mumbwa']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, gma in enumerate(key_gmas):
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        
        gma_data = df_clean[df_clean['GMA'] == gma]
        forest_data = gma_data[gma_data['Land Cover Class'] == 'Forest']
        cropland_data = gma_data[gma_data['Land Cover Class'] == 'Cropland']
        
        forest_pcts = []
        cropland_pcts = []
        
        for year in Config.YEARS:
            pct_col = f'{year} (%)'
            if not forest_data.empty and pct_col in forest_data.columns:
                forest_pcts.append(forest_data[pct_col].values[0])
            else:
                forest_pcts.append(0)
            
            if not cropland_data.empty and pct_col in cropland_data.columns:
                cropland_pcts.append(cropland_data[pct_col].values[0])
            else:
                cropland_pcts.append(0)
        
        line1 = ax1.plot(Config.YEARS, forest_pcts, 'o-', color='#228B22',
                        linewidth=3, markersize=8, label='Forest')
        line2 = ax2.plot(Config.YEARS, cropland_pcts, 's-', color='#FFD700',
                        linewidth=3, markersize=8, label='Cropland')
        
        ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Forest Cover (%)', fontsize=11, fontweight='bold', color='#228B22')
        ax2.set_ylabel('Cropland (%)', fontsize=11, fontweight='bold', color='#FFD700')
        
        ax1.tick_params(axis='y', labelcolor='#228B22')
        ax2.tick_params(axis='y', labelcolor='#FFD700')
        
        ax1.set_title(f'{gma} GMA', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=10)
    
    plt.suptitle('Forest Decline vs Cropland Expansion - Key GMAs',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_forest_vs_cropland_dual_axis.png')
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("GMA LAND COVER CHANGE VISUALIZATION SUITE")
    print("="*70)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    try:
        df, df_clean, gmas = load_and_prepare_data(Config.INPUT_FILE)
        
        successful = []
        failed = []
        
        # Generate visualizations
        try:
            plot_natural_vs_disturbed_trends(df, Config.OUTPUT_DIR)
            successful.append("fig_natural_vs_disturbed_trends.png")
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append("Natural vs Disturbed Trends")
        
        try:
            plot_composition_comparison(df_clean, Config.OUTPUT_DIR)
            successful.append("fig_composition_1984_vs_2024.png")
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append("Composition Comparison")
        
        try:
            plot_area_change_heatmap(df_clean, Config.OUTPUT_DIR)
            successful.append("fig_heatmap_area_change.png")
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append("Area Change Heatmap")
        
        try:
            plot_forest_trajectory(df_clean, Config.OUTPUT_DIR)
            successful.append("fig_forest_trajectory_slope.png")
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append("Forest Trajectory")
        
        try:
            plot_habitat_waterfall(df, Config.OUTPUT_DIR)
            successful.append("fig_waterfall_habitat_conversion.png")
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append("Habitat Waterfall")
        
        try:
            plot_forest_cropland_dual_axis(df_clean, Config.OUTPUT_DIR)
            successful.append("fig_forest_vs_cropland_dual_axis.png")
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append("Forest vs Cropland")
        
        # Summary
        print("\n" + "="*70)
        print("VISUALIZATION GENERATION COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {Config.OUTPUT_DIR}")
        
        if successful:
            print(f"\nSuccessfully generated {len(successful)} figure(s):")
            for i, fig in enumerate(successful, 1):
                print(f"  {i}. {fig}")
        
        if failed:
            print(f"\nFailed to generate {len(failed)} visualization(s):")
            for i, fig in enumerate(failed, 1):
                print(f"  {i}. {fig}")
        
    except FileNotFoundError:
        print(f"\nError: Could not find input file: {Config.INPUT_FILE}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
