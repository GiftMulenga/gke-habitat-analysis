"""
================================================================================
Data Acquisition and Preprocessing - Google Earth Engine
================================================================================

Objective B: Drivers of Habitat Loss Analysis - Greater Kafue Ecosystem

This script acquires and preprocesses driver variables using Google Earth Engine:
    - Proximity variables (distance to roads, settlements, rivers, KNP)
    - Socio-economic variables (population density, cultivation percentage)
    - Conservation variables (protection status)
    - Topographic variables (elevation, slope, aspect, TWI)
    - Climatic variables (rainfall, temperature)

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - earthengine-api
    - geemap
    - geopandas
    - numpy

================================================================================
"""

import ee
import geemap
import geopandas as gpd
import os
from datetime import datetime
import json
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for driver variable acquisition."""
    
    STUDY_AREA_NAME = "Greater Kafue Ecosystem"
    TARGET_CRS = "EPSG:32735"  # UTM Zone 35S
    TARGET_RESOLUTION = 30  # meters
    
    # Directory paths - UPDATE THESE
    BASE_DIR = "D:/Publication/Habitat Loss in the Greater Kafue Ecosystem"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = "E:/Research/Msc_Tropical_Ecology/GKE_Objective_B/data"
    
    # Input file paths
    BOUNDARIES = {
        "gke": os.path.join(DATA_DIR, "GKE_Boundary.shp"),
        "knp": os.path.join(DATA_DIR, "nationalParks.shp"),
        "gma": os.path.join(DATA_DIR, "GMA.shp"),
        "roads": os.path.join(DATA_DIR, "Roads.shp"),
    }
    
    # Export configuration
    EXPORT_FOLDER = "GKE_Driver_Variables"
    EXPORT_SCALE = 30
    MAX_PIXELS = 1e13
    
    # Driver variables by category
    DRIVER_VARIABLES = {
        "proximity": ["dist_roads", "dist_settlements", "dist_rivers", "dist_knp"],
        "socioeconomic": ["pop_density", "pop_change", "pct_cultivated"],
        "conservation": ["protection_status"],
        "topographic": ["elevation", "slope", "aspect", "twi"],
        "climatic": ["mean_rainfall", "mean_temp"]
    }


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        print("✓ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"⚠ GEE initialization failed: {e}")
        return False


def get_study_area():
    """Load study area boundary."""
    gke_path = Config.BOUNDARIES.get("gke")
    
    if gke_path and os.path.exists(gke_path):
        gdf = gpd.read_file(gke_path)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        geojson = json.loads(gdf.geometry.to_json())
        coords = []
        for feature in geojson['features']:
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords.extend(geom['coordinates'][0])
            elif geom['type'] == 'MultiPolygon':
                for polygon in geom['coordinates']:
                    coords.extend(polygon[0])
        
        study_area = ee.Geometry.Polygon(coords)
        area_km2 = study_area.area().divide(1e6).getInfo()
        print(f"  Study area: {area_km2:.2f} km²")
        return study_area
    
    raise FileNotFoundError(f"GKE boundary not found: {gke_path}")


def get_proximity_variables(study_area, boundaries):
    """Calculate proximity variables."""
    print("\n--- Calculating Proximity Variables ---")
    variables = {}
    
    if 'roads' in boundaries:
        dist_roads = boundaries['roads'].distance(maxError=100).clip(study_area)
        variables['dist_roads'] = dist_roads.rename('dist_roads')
        print("  ✓ dist_roads")
    
    water = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
    water_filtered = water.filterBounds(study_area)
    dist_rivers = water_filtered.distance(maxError=100).clip(study_area)
    variables['dist_rivers'] = dist_rivers.rename('dist_rivers')
    print("  ✓ dist_rivers")
    
    if 'knp' in boundaries:
        dist_knp = boundaries['knp'].distance(maxError=100).clip(study_area)
        variables['dist_knp'] = dist_knp.rename('dist_knp')
        print("  ✓ dist_knp")
    
    return variables


def get_socioeconomic_variables(study_area):
    """Calculate socio-economic variables."""
    print("\n--- Calculating Socio-economic Variables ---")
    variables = {}
    
    pop_2020 = ee.Image("WorldPop/GP/100m/pop/ZMB_2020").clip(study_area)
    pop_2000 = ee.Image("WorldPop/GP/100m/pop/ZMB_2000").clip(study_area)
    
    pop_density = pop_2020.resample('bilinear').reproject(
        crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['pop_density'] = pop_density.rename('pop_density')
    print("  ✓ pop_density")
    
    pop_change = pop_2020.subtract(pop_2000).divide(pop_2000.add(0.001)).multiply(100)
    pop_change = pop_change.resample('bilinear').reproject(
        crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['pop_change'] = pop_change.rename('pop_change')
    print("  ✓ pop_change")
    
    worldcover = ee.Image("ESA/WorldCover/v200/2021").clip(study_area)
    cropland = worldcover.eq(40)
    kernel = ee.Kernel.circle(radius=500, units='meters')
    pct_cultivated = cropland.reduceNeighborhood(
        reducer=ee.Reducer.mean(), kernel=kernel
    ).multiply(100).reproject(crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['pct_cultivated'] = pct_cultivated.rename('pct_cultivated')
    print("  ✓ pct_cultivated")
    
    return variables


def get_topographic_variables(study_area):
    """Calculate topographic variables."""
    print("\n--- Calculating Topographic Variables ---")
    variables = {}
    
    dem = ee.Image("USGS/SRTMGL1_003").clip(study_area)
    
    elevation = dem.reproject(crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['elevation'] = elevation.rename('elevation')
    print("  ✓ elevation")
    
    slope = ee.Terrain.slope(dem).reproject(crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['slope'] = slope.rename('slope')
    print("  ✓ slope")
    
    aspect = ee.Terrain.aspect(dem).reproject(crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['aspect'] = aspect.rename('aspect')
    print("  ✓ aspect")
    
    flow_acc = ee.Terrain.fillMinima(dem).subtract(dem).abs().add(1)
    slope_rad = slope.multiply(np.pi / 180)
    tan_slope = slope_rad.tan().add(0.001)
    twi = flow_acc.log().divide(tan_slope)
    twi = twi.reproject(crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['twi'] = twi.rename('twi')
    print("  ✓ twi")
    
    return variables


def get_climatic_variables(study_area):
    """Calculate climatic variables."""
    print("\n--- Calculating Climatic Variables ---")
    variables = {}
    
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate('2014-01-01', '2024-12-31') \
        .filterBounds(study_area)
    annual_rainfall = chirps.sum().divide(10)
    mean_rainfall = annual_rainfall.reproject(crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['mean_rainfall'] = mean_rainfall.rename('mean_rainfall')
    print("  ✓ mean_rainfall")
    
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
        .filterDate('2014-01-01', '2024-12-31') \
        .filterBounds(study_area) \
        .select('temperature_2m')
    mean_temp = era5.mean().subtract(273.15)
    mean_temp = mean_temp.resample('bilinear').reproject(
        crs=Config.TARGET_CRS, scale=Config.TARGET_RESOLUTION)
    variables['mean_temp'] = mean_temp.rename('mean_temp')
    print("  ✓ mean_temp")
    
    return variables


def export_all_variables(all_variables, study_area):
    """Export all variables to Google Drive."""
    tasks = {}
    
    for category, variables in all_variables.items():
        for var_name, image in variables.items():
            try:
                task = ee.batch.Export.image.toDrive(
                    image=image,
                    description=var_name,
                    folder=f"{Config.EXPORT_FOLDER}/{category}",
                    region=study_area,
                    scale=Config.EXPORT_SCALE,
                    crs=Config.TARGET_CRS,
                    maxPixels=Config.MAX_PIXELS,
                    fileFormat='GeoTIFF'
                )
                task.start()
                tasks[var_name] = task
                print(f"  ✓ Started export: {var_name}")
            except Exception as e:
                print(f"  ✗ Export failed for {var_name}: {e}")
    
    return tasks


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("DATA ACQUISITION - Google Earth Engine")
    print("Greater Kafue Ecosystem - Driver Variables")
    print("="*60)
    
    if not initialize_gee():
        return None
    
    study_area = get_study_area()
    
    # Load boundaries
    boundaries = {}
    for name, path in Config.BOUNDARIES.items():
        if path and os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                geojson = json.loads(gdf.geometry.to_json())
                fc = ee.FeatureCollection(geojson['features'])
                boundaries[name] = fc
            except Exception as e:
                print(f"  ⚠ {name}: {e}")
    
    # Process variables
    all_variables = {}
    all_variables['proximity'] = get_proximity_variables(study_area, boundaries)
    all_variables['socioeconomic'] = get_socioeconomic_variables(study_area)
    all_variables['topographic'] = get_topographic_variables(study_area)
    all_variables['climatic'] = get_climatic_variables(study_area)
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    
    total = sum(len(v) for v in all_variables.values())
    print(f"  Total variables processed: {total}")
    
    # Export
    tasks = export_all_variables(all_variables, study_area)
    print(f"\n  Monitor at: https://code.earthengine.google.com/tasks")
    
    return {'variables': all_variables, 'tasks': tasks, 'study_area': study_area}


if __name__ == "__main__":
    results = main()
