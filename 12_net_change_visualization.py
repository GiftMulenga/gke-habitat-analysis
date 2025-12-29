"""
================================================================================
Decadal Net Change Visualization
================================================================================

Publication-quality visualization of decadal land cover net change
in the Greater Kafue Ecosystem (1984-2024).

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - pandas
    - matplotlib

================================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for net change visualization."""
    
    # Input paths - UPDATE THESE
    PATHS = {
        "1984-1994": "D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/tables/Table_Net_Change_GKE_1984_1994.csv",
        "1994-2004": "D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/tables/Table_Net_Change_GKE_1994_2004.csv",
        "2004-2014": "D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/tables/Table_Net_Change_GKE_2004_2014.csv",
        "2014-2024": "D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/tables/Table_Net_Change_GKE_2014_2024.csv"
    }
    
    OUTPUT_DIR = "D:/Publication/Habitat Loss in the Greater Kafue Ecosystem/outputs/objective_a/figures"
    
    DPI = 300
    
    # Color scheme
    COLORS = {
        'Built-up': '#E74C3C',
        'Forest': '#27AE60',
        'Cropland': '#F39C12',
        'Grassland': '#95A5A6',
        'Bareland': '#8B4513',
        'Water': '#3498DB'
    }


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_net_change_data():
    """
    Load and combine net change data from all decades.
    
    Returns:
        DataFrame: Combined net change data
    """
    print("Loading net change data...")
    
    dfs = []
    for period, path in Config.PATHS.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[["Land Cover Class", "Net Change (ha)"]]
            df = df.rename(columns={"Net Change (ha)": period})
            dfs.append(df)
            print(f"  Loaded: {period}")
        else:
            print(f"  Warning: {path} not found")
    
    if not dfs:
        return None
    
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.merge(df, on="Land Cover Class")
    
    # Calculate total
    combined["Total 1984-2024"] = combined.iloc[:, 1:].sum(axis=1)
    
    return combined


def create_stacked_bar_chart(combined_df, output_path):
    """
    Create stacked bar chart of decadal net change.
    
    Args:
        combined_df: Combined net change DataFrame
        output_path: Path to save figure
    """
    print("\nCreating stacked bar chart...")
    
    # Prepare data
    plot_df = combined_df.set_index("Land Cover Class").iloc[:, 0:4]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get colors
    colors = [Config.COLORS.get(lc, 'gray') for lc in plot_df.index]
    
    plot_df.T.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel("Decade", fontsize=12, fontweight='bold')
    ax.set_ylabel("Net Change (ha)", fontsize=12, fontweight='bold')
    ax.set_title("Decadal Net Land Cover Change in the GKE (1984-2024)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Land Cover Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_grouped_bar_chart(combined_df, output_path):
    """
    Create grouped bar chart of decadal net change.
    
    Args:
        combined_df: Combined net change DataFrame
        output_path: Path to save figure
    """
    print("\nCreating grouped bar chart...")
    
    plot_df = combined_df.set_index("Land Cover Class").iloc[:, 0:4]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = range(len(plot_df.columns))
    width = 0.12
    
    for i, (lc_class, values) in enumerate(plot_df.iterrows()):
        offset = (i - len(plot_df) / 2 + 0.5) * width
        color = Config.COLORS.get(lc_class, 'gray')
        ax.bar([xi + offset for xi in x], values, width, label=lc_class, color=color, edgecolor='white')
    
    ax.set_xlabel("Decade", fontsize=12, fontweight='bold')
    ax.set_ylabel("Net Change (ha)", fontsize=12, fontweight='bold')
    ax.set_title("Decadal Net Land Cover Change by Class (1984-2024)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.columns)
    ax.legend(title='Land Cover Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_total_change_chart(combined_df, output_path):
    """
    Create horizontal bar chart of total change (1984-2024).
    
    Args:
        combined_df: Combined net change DataFrame
        output_path: Path to save figure
    """
    print("\nCreating total change chart...")
    
    total_df = combined_df[['Land Cover Class', 'Total 1984-2024']].copy()
    total_df = total_df.sort_values('Total 1984-2024')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#E74C3C' if v < 0 else '#27AE60' for v in total_df['Total 1984-2024']]
    
    ax.barh(total_df['Land Cover Class'], total_df['Total 1984-2024'], color=colors, edgecolor='black')
    
    ax.set_xlabel("Net Change (ha)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Land Cover Class", fontsize=12, fontweight='bold')
    ax.set_title("Total Net Land Cover Change (1984-2024)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (idx, row) in enumerate(total_df.iterrows()):
        value = row['Total 1984-2024']
        ax.text(value + (5000 if value >= 0 else -5000), i,
               f'{value:,.0f}', va='center',
               ha='left' if value >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("DECADAL NET CHANGE VISUALIZATION")
    print("="*60)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    combined_df = load_net_change_data()
    
    if combined_df is None:
        print("\nError: No data loaded.")
        return
    
    # Display summary
    print("\nNet Change Summary (ha):")
    print(combined_df.to_string(index=False))
    
    # Create visualizations
    create_stacked_bar_chart(
        combined_df,
        os.path.join(Config.OUTPUT_DIR, 'fig_decadal_net_change_stacked.png')
    )
    
    create_grouped_bar_chart(
        combined_df,
        os.path.join(Config.OUTPUT_DIR, 'fig_decadal_net_change_grouped.png')
    )
    
    create_total_change_chart(
        combined_df,
        os.path.join(Config.OUTPUT_DIR, 'fig_total_net_change_1984_2024.png')
    )
    
    # Save summary table
    combined_df.to_csv(
        os.path.join(Config.OUTPUT_DIR, 'Table_Decadal_Net_Change_Summary.csv'),
        index=False
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print(f"Outputs saved to: {Config.OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
