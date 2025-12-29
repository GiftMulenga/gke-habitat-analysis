"""
================================================================================
GKE Habitat Analysis - Objective A
================================================================================

Historical Habitat Dynamics in the Greater Kafue Ecosystem (1984-2024)

This module performs comprehensive habitat change analysis including:
    1. Land Cover Area Statistics by Management Zone
    2. Change Detection and Transition Matrices
    3. KNP Boundary Gradient Analysis
    4. GMA Centroid Gradient Analysis
    5. Road Proximity Analysis

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - rasterio
    - geopandas
    - pandas
    - numpy
    - matplotlib
    - scipy
    - shapely

================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping, Point
from shapely.ops import unary_union
from shapely.validation import make_valid
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import glob
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for the GKE habitat analysis."""
    
    # Land Cover Classification Scheme (6 Classes)
    CLASS_LABELS = {
        1: "Built-up",
        2: "Forest",
        3: "Cropland",
        4: "Grassland",
        5: "Bareland",
        6: "Water"
    }
    
    CLASS_COLORS = {
        1: "#FF0000",  # Red - Built-up
        2: "#008000",  # Green - Forest
        3: "#FFFF00",  # Yellow - Cropland
        4: "#90EE90",  # Light Green - Grassland
        5: "#FFB347",  # Peach - Bareland
        6: "#0000FF",  # Blue - Water
    }
    
    # Habitat classification
    NATURAL_HABITAT_CLASSES = [2, 4, 6]  # Forest, Grassland, Water
    DISTURBED_CLASSES = [1, 3, 5]  # Built-up, Cropland, Bareland
    FOREST_CLASSES = [2]
    
    # Spatial resolution
    CELL_SIZE = 30  # meters (Landsat resolution)
    PIXEL_TO_HA = (CELL_SIZE * CELL_SIZE) / 10000  # 0.09 ha per pixel
    
    # Coordinate Reference System
    CRS_EPSG = 32735  # UTM Zone 35S
    
    # Time points
    YEARS = [1984, 1994, 2004, 2014, 2024]
    
    # Gradient analysis parameters
    KNP_BUFFER_INCREMENT = 200  # meters
    KNP_MAX_BUFFER_DISTANCE = 15000  # meters (15 km)
    GMA_BUFFER_INCREMENT = 200  # meters
    GMA_MAX_BUFFER_DISTANCE = 10000  # meters (10 km)
    ROAD_BUFFER_INCREMENT = 200  # meters
    ROAD_MAX_BUFFER_DISTANCE = 10000  # meters (10 km)
    
    # Road classification mapping
    ROAD_CLASS_MAPPING = {
        "primary": "Primary",
        "trunk": "Primary",
        "secondary": "Secondary",
        "tertiary": "Tertiary",
        "residential": "Tertiary",
        "unclassified": "Tertiary",
    }


# =============================================================================
# GEOMETRY HELPER FUNCTIONS
# =============================================================================

def fix_geometry(gdf):
    """
    Fix invalid geometries in a GeoDataFrame.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with potentially invalid geometries
    
    Returns:
        tuple: (Fixed GeoDataFrame, Number of fixes applied)
    """
    fixes_count = 0
    
    # Remove null geometries
    null_mask = gdf.geometry.isna()
    if null_mask.any():
        fixes_count += null_mask.sum()
        gdf = gdf[~null_mask].copy()
    
    # Fix invalid geometries using buffer(0) method
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        fixes_count += invalid_mask.sum()
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
        
        # Apply make_valid for any still invalid
        still_invalid = ~gdf.geometry.is_valid
        if still_invalid.any():
            gdf.loc[still_invalid, 'geometry'] = gdf.loc[still_invalid, 'geometry'].apply(
                lambda geom: make_valid(geom) if not geom.is_valid else geom
            )
    
    # Remove duplicate geometries
    duplicate_mask = gdf.geometry.duplicated()
    if duplicate_mask.any():
        fixes_count += duplicate_mask.sum()
        gdf = gdf[~duplicate_mask].copy()
    
    return gdf, fixes_count


def calculate_area_ha(gdf, target_crs_epsg=32735):
    """
    Calculate total area in hectares.
    
    Args:
        gdf (GeoDataFrame): Input GeoDataFrame
        target_crs_epsg (int): EPSG code for projected CRS
    
    Returns:
        float: Total area in hectares
    """
    if gdf.crs and gdf.crs.to_epsg() != target_crs_epsg:
        gdf = gdf.to_crs(epsg=target_crs_epsg)
    elif not gdf.crs:
        gdf = gdf.set_crs(epsg=target_crs_epsg)
    
    return gdf.geometry.area.sum() / 10000


# =============================================================================
# AREA CALCULATION FUNCTIONS
# =============================================================================

def pixels_to_hectares(pixel_count):
    """Convert pixel count to hectares."""
    return pixel_count * Config.PIXEL_TO_HA


def load_boundary(path, filter_name=None, name_field="NAME"):
    """
    Load a boundary shapefile and optionally filter by name.
    
    Args:
        path (str): Path to shapefile
        filter_name (str): Optional name to filter by
        name_field (str): Column name containing feature names
        
    Returns:
        GeoDataFrame: Loaded and filtered boundary
    """
    gdf = gpd.read_file(path)
    if gdf.crs.to_epsg() != Config.CRS_EPSG:
        gdf = gdf.to_crs(epsg=Config.CRS_EPSG)
    if filter_name and name_field in gdf.columns:
        gdf = gdf[gdf[name_field].str.contains(filter_name, case=False, na=False)]
    return gdf


def calculate_class_areas(raster_path, geometry=None):
    """
    Calculate area for each land cover class within a geometry.
    
    Args:
        raster_path (str): Path to the LULC raster
        geometry (GeoDataFrame): Optional geometry to mask the raster
    
    Returns:
        dict: {class_value: area_in_hectares}
    """
    with rasterio.open(raster_path) as src:
        if geometry is not None:
            geoms = [mapping(geom) for geom in geometry.geometry]
            out_image, _ = mask(src, geoms, crop=True, nodata=0)
            data = out_image[0]
        else:
            data = src.read(1)
        
        unique, counts = np.unique(data[data > 0], return_counts=True)
        class_areas = {int(c): pixels_to_hectares(n) for c, n in zip(unique, counts)}
        
    return class_areas


def create_area_table(class_areas_dict, years, class_labels):
    """
    Create a formatted area table from class areas dictionary.
    
    Args:
        class_areas_dict (dict): Areas by year and class
        years (list): Years to include
        class_labels (dict): Class value to label mapping
        
    Returns:
        DataFrame: Formatted area table
    """
    data = []
    
    for class_val, label in class_labels.items():
        row = {"Land Cover Class": label}
        for year in years:
            area = class_areas_dict.get(year, {}).get(class_val, 0)
            total = sum(class_areas_dict.get(year, {}).values())
            pct = (area / total * 100) if total > 0 else 0
            row[f"{year} Area (ha)"] = round(area, 2)
            row[f"{year} (%)"] = round(pct, 2)
        data.append(row)
    
    # Add Total row
    total_row = {"Land Cover Class": "Total"}
    for year in years:
        total = sum(class_areas_dict.get(year, {}).values())
        total_row[f"{year} Area (ha)"] = round(total, 2)
        total_row[f"{year} (%)"] = 100.0
    data.append(total_row)
    
    # Add Natural Habitat row
    natural_row = {"Land Cover Class": "Natural Habitat*"}
    for year in years:
        natural_area = sum(class_areas_dict.get(year, {}).get(c, 0) 
                         for c in Config.NATURAL_HABITAT_CLASSES)
        total = sum(class_areas_dict.get(year, {}).values())
        pct = (natural_area / total * 100) if total > 0 else 0
        natural_row[f"{year} Area (ha)"] = round(natural_area, 2)
        natural_row[f"{year} (%)"] = round(pct, 2)
    data.append(natural_row)
    
    # Add Disturbed row
    disturbed_row = {"Land Cover Class": "Disturbed†"}
    for year in years:
        disturbed_area = sum(class_areas_dict.get(year, {}).get(c, 0) 
                           for c in Config.DISTURBED_CLASSES)
        total = sum(class_areas_dict.get(year, {}).values())
        pct = (disturbed_area / total * 100) if total > 0 else 0
        disturbed_row[f"{year} Area (ha)"] = round(disturbed_area, 2)
        disturbed_row[f"{year} (%)"] = round(pct, 2)
    data.append(disturbed_row)
    
    return pd.DataFrame(data)


def analyze_zone(zone_name, geometry, raster_paths, years):
    """
    Analyze land cover composition for a specific zone.
    
    Args:
        zone_name (str): Name of the zone
        geometry (GeoDataFrame): Zone boundary
        raster_paths (dict): LULC raster paths by year
        years (list): Years to analyze
        
    Returns:
        DataFrame: Land cover composition table
    """
    print(f"  Analyzing {zone_name}...")
    
    class_areas_dict = {}
    for year in years:
        if year in raster_paths:
            areas = calculate_class_areas(raster_paths[year], geometry)
            class_areas_dict[year] = areas
    
    df = create_area_table(class_areas_dict, [y for y in years if y in raster_paths], 
                          Config.CLASS_LABELS)
    return df


# =============================================================================
# CHANGE DETECTION FUNCTIONS
# =============================================================================

def calculate_transition_matrix(raster_t1_path, raster_t2_path, geometry=None):
    """
    Calculate transition matrix between two time periods.
    
    Args:
        raster_t1_path (str): Path to time 1 raster
        raster_t2_path (str): Path to time 2 raster
        geometry (GeoDataFrame): Optional geometry to mask
        
    Returns:
        DataFrame: Transition matrix with areas in hectares
    """
    with rasterio.open(raster_t1_path) as src_t1:
        if geometry is not None:
            geoms = [mapping(geom) for geom in geometry.geometry]
            data_t1, _ = mask(src_t1, geoms, crop=True, nodata=0)
            data_t1 = data_t1[0]
        else:
            data_t1 = src_t1.read(1)
    
    with rasterio.open(raster_t2_path) as src_t2:
        if geometry is not None:
            geoms = [mapping(geom) for geom in geometry.geometry]
            data_t2, _ = mask(src_t2, geoms, crop=True, nodata=0)
            data_t2 = data_t2[0]
        else:
            data_t2 = src_t2.read(1)
    
    # Ensure same shape
    if data_t1.shape != data_t2.shape:
        min_rows = min(data_t1.shape[0], data_t2.shape[0])
        min_cols = min(data_t1.shape[1], data_t2.shape[1])
        data_t1 = data_t1[:min_rows, :min_cols]
        data_t2 = data_t2[:min_rows, :min_cols]
    
    # Calculate transitions
    classes = list(Config.CLASS_LABELS.keys())
    matrix = np.zeros((len(classes), len(classes)))
    
    for i, class_from in enumerate(classes):
        for j, class_to in enumerate(classes):
            count = np.sum((data_t1 == class_from) & (data_t2 == class_to))
            matrix[i, j] = pixels_to_hectares(count)
    
    # Create DataFrame
    df = pd.DataFrame(
        matrix,
        index=[Config.CLASS_LABELS[c] for c in classes],
        columns=[Config.CLASS_LABELS[c] for c in classes]
    )
    
    return df


def calculate_net_change(class_areas_dict, years):
    """
    Calculate net change in area for each class between years.
    
    Args:
        class_areas_dict (dict): Areas by year and class
        years (list): Years to analyze
        
    Returns:
        DataFrame: Net change summary
    """
    data = []
    
    for class_val, label in Config.CLASS_LABELS.items():
        row = {"Land Cover Class": label}
        
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i+1]
            area_t1 = class_areas_dict.get(y1, {}).get(class_val, 0)
            area_t2 = class_areas_dict.get(y2, {}).get(class_val, 0)
            change = area_t2 - area_t1
            pct_change = (change / area_t1 * 100) if area_t1 > 0 else 0
            
            row[f"{y1}-{y2} Change (ha)"] = round(change, 2)
            row[f"{y1}-{y2} Change (%)"] = round(pct_change, 2)
        
        # Total change
        first_area = class_areas_dict.get(years[0], {}).get(class_val, 0)
        last_area = class_areas_dict.get(years[-1], {}).get(class_val, 0)
        total_change = last_area - first_area
        total_pct = (total_change / first_area * 100) if first_area > 0 else 0
        
        row[f"{years[0]}-{years[-1]} Total (ha)"] = round(total_change, 2)
        row[f"{years[0]}-{years[-1]} Total (%)"] = round(total_pct, 2)
        
        data.append(row)
    
    return pd.DataFrame(data)


# =============================================================================
# GRADIENT ANALYSIS FUNCTIONS
# =============================================================================

def create_distance_raster(boundary_gdf, reference_raster_path, output_path=None):
    """
    Create a distance raster from a boundary.
    
    Args:
        boundary_gdf (GeoDataFrame): Boundary geometry
        reference_raster_path (str): Reference raster for extent/resolution
        output_path (str): Optional path to save raster
        
    Returns:
        tuple: (distance array, profile)
    """
    with rasterio.open(reference_raster_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)
    
    # Rasterize boundary
    boundary_raster = rasterize(
        [(geom, 1) for geom in boundary_gdf.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    
    # Calculate distance
    distance = distance_transform_edt(boundary_raster == 0)
    distance = distance * profile['transform'][0]  # Convert to meters
    
    if output_path:
        profile.update(dtype=np.float32, count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(distance.astype(np.float32), 1)
    
    return distance, profile


def analyze_gradient(lulc_raster_path, distance_raster, max_distance, increment):
    """
    Analyze land cover composition at different distances from a boundary.
    
    Args:
        lulc_raster_path (str): Path to LULC raster
        distance_raster (ndarray): Distance values
        max_distance (int): Maximum distance to analyze
        increment (int): Distance increment
        
    Returns:
        DataFrame: Gradient analysis results
    """
    with rasterio.open(lulc_raster_path) as src:
        lulc = src.read(1)
    
    # Ensure same shape
    if lulc.shape != distance_raster.shape:
        min_rows = min(lulc.shape[0], distance_raster.shape[0])
        min_cols = min(lulc.shape[1], distance_raster.shape[1])
        lulc = lulc[:min_rows, :min_cols]
        distance_raster = distance_raster[:min_rows, :min_cols]
    
    results = []
    distances = list(range(0, max_distance + increment, increment))
    
    for dist in distances:
        # Create distance band mask
        if dist == 0:
            band_mask = distance_raster <= increment
        else:
            band_mask = (distance_raster > dist) & (distance_raster <= dist + increment)
        
        band_data = lulc[band_mask]
        total_pixels = len(band_data[band_data > 0])
        
        if total_pixels == 0:
            continue
        
        row = {"Distance_m": dist + increment}
        
        # Calculate percentage for each class
        for class_val, label in Config.CLASS_LABELS.items():
            count = np.sum(band_data == class_val)
            pct = (count / total_pixels * 100) if total_pixels > 0 else 0
            row[f"{label}_%"] = round(pct, 2)
        
        # Calculate aggregated categories
        natural_count = sum(np.sum(band_data == c) for c in Config.NATURAL_HABITAT_CLASSES)
        disturbed_count = sum(np.sum(band_data == c) for c in Config.DISTURBED_CLASSES)
        forest_count = sum(np.sum(band_data == c) for c in Config.FOREST_CLASSES)
        
        row["Natural_Habitat_%"] = round(natural_count / total_pixels * 100, 2)
        row["Disturbed_%"] = round(disturbed_count / total_pixels * 100, 2)
        row["Forest_%"] = round(forest_count / total_pixels * 100, 2)
        row["Total_Pixels"] = total_pixels
        
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_temporal_trends(class_trends, years, output_path):
    """
    Create temporal trend line plot for land cover classes.
    
    Args:
        class_trends (dict): Trends by class
        years (list): Years
        output_path (str): Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for class_val, label in Config.CLASS_LABELS.items():
        if label in class_trends:
            ax.plot(years, class_trends[label], 
                   marker='o', linewidth=2.5, markersize=8,
                   color=Config.CLASS_COLORS[class_val], label=label)
    
    ax.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax.set_ylabel("Coverage (%)", fontsize=12, fontweight='bold')
    ax.set_title("Temporal Trends in Land Cover Composition", 
                fontsize=14, fontweight='bold')
    ax.legend(title="Land Cover Class", fontsize=10, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(years)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")


def plot_gradient_analysis(gradient_df, output_path, title="Gradient Analysis"):
    """
    Create gradient analysis visualization.
    
    Args:
        gradient_df (DataFrame): Gradient analysis results
        output_path (str): Output file path
        title (str): Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: All individual classes
    for class_val, label in Config.CLASS_LABELS.items():
        col_name = f"{label}_%"
        if col_name in gradient_df.columns:
            ax1.plot(gradient_df["Distance_m"], gradient_df[col_name], 
                    marker='o', linewidth=2, label=label,
                    color=Config.CLASS_COLORS[class_val])
    
    ax1.set_xlabel("Distance (m)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coverage (%)", fontsize=11, fontweight='bold')
    ax1.set_title(f"{title} - Individual Classes", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Aggregated categories
    ax2.plot(gradient_df["Distance_m"], gradient_df["Natural_Habitat_%"], 
            marker='o', linewidth=3, label='Natural Habitat', color='#006400')
    ax2.plot(gradient_df["Distance_m"], gradient_df["Disturbed_%"], 
            marker='s', linewidth=3, label='Disturbed', color='#8B0000')
    ax2.plot(gradient_df["Distance_m"], gradient_df["Forest_%"], 
            marker='^', linewidth=2, label='Forest', color='#228B22')
    
    ax2.set_xlabel("Distance (m)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coverage (%)", fontsize=11, fontweight='bold')
    ax2.set_title(f"{title} - Aggregated Categories", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")


# =============================================================================
# OUTPUT HELPER FUNCTIONS
# =============================================================================

def get_output_path(output_dir, filename, subdir="tables"):
    """
    Get full output path for a file.
    
    Args:
        output_dir (str): Base output directory
        filename (str): File name
        subdir (str): Subdirectory (tables, figures, rasters)
        
    Returns:
        str: Full output path
    """
    full_dir = os.path.join(output_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)
    return os.path.join(full_dir, filename)


# =============================================================================
# MAIN ANALYSIS WORKFLOW
# =============================================================================

def run_objective_a_analysis(data_dir, output_dir, lulc_rasters, boundaries):
    """
    Run complete Objective A analysis.
    
    Args:
        data_dir (str): Input data directory
        output_dir (str): Output directory
        lulc_rasters (dict): LULC raster paths by year
        boundaries (dict): Boundary shapefile paths
        
    Returns:
        dict: Analysis results
    """
    print("="*70)
    print("GKE OBJECTIVE A ANALYSIS")
    print("Historical Habitat Dynamics (1984-2024)")
    print("="*70)
    
    # Create output directories
    for subdir in ['tables', 'figures', 'rasters']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    results = {}
    
    # Check input files
    print("\nChecking input data files...")
    rasters_found = {}
    for year, path in lulc_rasters.items():
        if os.path.exists(path):
            rasters_found[year] = path
            print(f"  ✓ {year}: {os.path.basename(path)}")
        else:
            print(f"  ✗ {year}: NOT FOUND")
    
    boundaries_found = {}
    for name, path in boundaries.items():
        if os.path.exists(path):
            boundaries_found[name] = path
            print(f"  ✓ {name}: {os.path.basename(path)}")
        else:
            print(f"  ✗ {name}: NOT FOUND")
    
    if len(rasters_found) == 0:
        print("\n⚠ WARNING: No rasters found! Please check file paths.")
        return None
    
    # 1. GKE-Wide Analysis
    if "gke" in boundaries_found:
        print("\n" + "="*50)
        print("GKE-WIDE LAND COVER ANALYSIS")
        print("="*50)
        
        gke_boundary = load_boundary(boundaries_found["gke"])
        df_gke = analyze_zone("GKE", gke_boundary, rasters_found, Config.YEARS)
        
        output_path = get_output_path(output_dir, "Table_GKE_Wide_Composition.csv")
        df_gke.to_csv(output_path, index=False)
        print(f"✓ Saved: {os.path.basename(output_path)}")
        
        results['gke_composition'] = df_gke
    
    # 2. KNP Analysis
    if "knp" in boundaries_found:
        print("\n" + "="*50)
        print("KAFUE NATIONAL PARK ANALYSIS")
        print("="*50)
        
        knp_boundary = load_boundary(boundaries_found["knp"])
        df_knp = analyze_zone("KNP", knp_boundary, rasters_found, Config.YEARS)
        
        output_path = get_output_path(output_dir, "Table_KNP_Composition.csv")
        df_knp.to_csv(output_path, index=False)
        print(f"✓ Saved: {os.path.basename(output_path)}")
        
        results['knp_composition'] = df_knp
    
    # 3. GMA Analysis
    if "gma" in boundaries_found:
        print("\n" + "="*50)
        print("GAME MANAGEMENT AREAS ANALYSIS")
        print("="*50)
        
        gma_boundary = load_boundary(boundaries_found["gma"])
        
        # Check for name column
        name_cols = ['NAME', 'GMA_NAME', 'Zone_Name', 'name']
        gma_name_col = None
        for col in name_cols:
            if col in gma_boundary.columns:
                gma_name_col = col
                break
        
        if gma_name_col:
            gma_names = gma_boundary[gma_name_col].unique()
            print(f"Found {len(gma_names)} GMAs")
            
            for gma_name in gma_names:
                gma_single = gma_boundary[gma_boundary[gma_name_col] == gma_name]
                df_gma = analyze_zone(gma_name, gma_single, rasters_found, Config.YEARS)
                
                safe_name = gma_name.replace(" ", "_").replace("/", "_")
                output_path = get_output_path(output_dir, f"Table_{safe_name}_Composition.csv")
                df_gma.to_csv(output_path, index=False)
        
        results['gma_analysis'] = True
    
    # 4. Transition Matrices
    print("\n" + "="*50)
    print("CHANGE DETECTION ANALYSIS")
    print("="*50)
    
    years_available = sorted(rasters_found.keys())
    
    for i in range(len(years_available) - 1):
        y1, y2 = years_available[i], years_available[i+1]
        print(f"\nCalculating transition matrix {y1} → {y2}...")
        
        trans_matrix = calculate_transition_matrix(
            rasters_found[y1], 
            rasters_found[y2]
        )
        
        output_path = get_output_path(output_dir, f"Table_Transition_{y1}_{y2}.csv")
        trans_matrix.to_csv(output_path)
        print(f"✓ Saved: {os.path.basename(output_path)}")
    
    results['transition_matrices'] = True
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    n_tables = len(glob.glob(os.path.join(output_dir, "tables", "*.csv")))
    n_figures = len(glob.glob(os.path.join(output_dir, "figures", "*.png")))
    n_rasters = len(glob.glob(os.path.join(output_dir, "rasters", "*.tif")))
    
    print(f"\nOutput Summary:")
    print(f"  Tables: {n_tables} CSV files")
    print(f"  Figures: {n_figures} PNG files")
    print(f"  Rasters: {n_rasters} TIF files")
    print(f"\nOutput directory: {output_dir}")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/objective_a")
    
    # Input file paths
    LULC_RASTERS = {
        1984: os.path.join(DATA_DIR, "GKE_1984.tif"),
        1994: os.path.join(DATA_DIR, "GKE_1994.tif"),
        2004: os.path.join(DATA_DIR, "GKE_2004.tif"),
        2014: os.path.join(DATA_DIR, "GKE_2014.tif"),
        2024: os.path.join(DATA_DIR, "GKE_2024.tif"),
    }
    
    BOUNDARIES = {
        "gke": os.path.join(DATA_DIR, "GKE_Boundary.shp"),
        "knp": os.path.join(DATA_DIR, "nationalParks.shp"),
        "gma": os.path.join(DATA_DIR, "GMA.shp"),
        "roads": os.path.join(DATA_DIR, "Roads.shp"),
    }
    
    # Run analysis
    results = run_objective_a_analysis(DATA_DIR, OUTPUT_DIR, LULC_RASTERS, BOUNDARIES)
