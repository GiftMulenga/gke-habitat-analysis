"""
================================================================================
GKE Landsat Processing, Visualization, and Export Script
================================================================================

Purpose: Process and visualize Landsat data for habitat loss analysis (1984-2024)
         in the Greater Kafue Ecosystem, Zambia.

Components:
    Part 1: GKELandsatVisualization - Process and visualize Landsat composites
    Part 2: GKELandsatExporter - Export composites to Google Drive

Author: Gift Mulenga
Institution: Copperbelt University, Zambia
Research: MSc Thesis - Tropical Ecology

Requirements:
    - earthengine-api
    - geemap
    - geopandas
    - pickle
    
================================================================================
"""

import ee
import geemap
import geopandas as gpd
import os
import json
import pickle
from datetime import datetime
import time


# =============================================================================
# LANDSAT VISUALIZATION CLASS
# =============================================================================

class GKELandsatVisualization:
    """
    Process and visualize Landsat data for the Greater Kafue Ecosystem.
    
    This class handles:
        - Loading study area boundaries from shapefiles
        - Retrieving appropriate Landsat collections by year
        - Creating cloud-free median composites
        - Interactive visualization with geemap
        - Saving processor state for export
    
    Attributes:
        shapefile_path (str): Path to GKE boundary shapefile
        output_folder (str): Local folder for saving processor state
        cloud_threshold (int): Maximum cloud cover percentage (0-100)
        target_configs (dict): Date ranges for each target year
        study_area: Earth Engine geometry of study area
        map: geemap Map instance
        composites (dict): Processed composites by year
        configs (dict): Configuration metadata by year
    """
    
    def __init__(self, shapefile_path, output_folder, cloud_threshold=1):
        """
        Initialize the GKE Landsat processor for visualization.
        
        Args:
            shapefile_path (str): Path to GKE boundary shapefile
            output_folder (str): Local folder for saving processor state
            cloud_threshold (int): Maximum cloud cover percentage (0-100)
        """
        self.shapefile_path = shapefile_path
        self.output_folder = output_folder
        self.study_area = None
        self.map = None
        
        # Target years and date ranges for dry season composites
        self.target_configs = {
            1984: {'start_date': '1984-05-01', 'end_date': '1984-08-31'},
            1994: {'start_date': '1994-05-01', 'end_date': '1994-08-31'},
            2004: {'start_date': '2004-05-01', 'end_date': '2004-08-31'},
            2014: {'start_date': '2014-05-01', 'end_date': '2014-08-31'},
            2024: {'start_date': '2024-05-01', 'end_date': '2024-06-30'}
        }
        
        self.cloud_threshold = cloud_threshold

    def update_date_range(self, year, start_date, end_date):
        """
        Update date range for a specific year.
        
        Args:
            year (int): Target year
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        if year in self.target_configs:
            self.target_configs[year]['start_date'] = start_date
            self.target_configs[year]['end_date'] = end_date
            print(f"Updated {year} date range: {start_date} to {end_date}")
        else:
            print(f"Year {year} not found in target configurations")
    
    def update_cloud_threshold(self, threshold):
        """
        Update cloud cover threshold.
        
        Args:
            threshold (int): Maximum cloud cover percentage (0-100)
        """
        self.cloud_threshold = threshold
        print(f"Updated cloud cover threshold to {threshold}%")
        
    def load_study_area(self):
        """
        Load GKE boundary shapefile and convert to Earth Engine geometry.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            gdf = gpd.read_file(self.shapefile_path)

            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')

            geojson = json.loads(gdf.to_json())
            coords = []
            for feature in geojson['features']:
                geom = feature['geometry']
                if geom['type'] == 'Polygon':
                    coords.extend(geom['coordinates'][0])
                elif geom['type'] == 'MultiPolygon':
                    for polygon in geom['coordinates']:
                        coords.extend(polygon[0])

            self.study_area = ee.Geometry.Polygon(coords)
            self.study_area_buffered = self.study_area.buffer(5000)
            
            print(f"Study area loaded successfully!")
            print(f"Area: {self.study_area.area().divide(1000000).getInfo():.2f} km²")
            return True
            
        except Exception as e:
            print(f"Error loading study area: {e}")
            return False
    
    def get_landsat_collection(self, year):
        """
        Get appropriate Landsat collection based on year.
        
        Args:
            year (int): Target year
            
        Returns:
            dict: Collection ID and band names
        """
        if year <= 2011:
            collection = 'LANDSAT/LT05/C02/T1_L2'
            bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
        else:
            collection = 'LANDSAT/LC08/C02/T1_L2'
            bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        return {'collection': collection, 'bands': bands}
    
    def scale_landsat(self, image, landsat_type):
        """
        Apply scaling factors to Landsat Collection 2 Level 2 surface reflectance.
        
        Args:
            image: Earth Engine image
            landsat_type (str): Landsat sensor type identifier
            
        Returns:
            ee.Image: Scaled image
        """
        if landsat_type in ['LT05']:
            optical_bands = image.select(
                ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
            ).multiply(0.0000275).add(-0.2)
        else:
            optical_bands = image.select(
                ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            ).multiply(0.0000275).add(-0.2)
        return image.addBands(optical_bands, None, True)
    
    def create_composite(self, year):
        """
        Create annual median composite for a specific year.
        
        Args:
            year (int): Target year
            
        Returns:
            tuple: (composite image, configuration dict) or (None, config) if no images
        """
        config = self.get_landsat_collection(year)
        start_date = self.target_configs[year]['start_date']
        end_date = self.target_configs[year]['end_date']
        
        print(f"Processing {year} using {config['collection']}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Cloud cover threshold: {self.cloud_threshold}%")
        
        collection = ee.ImageCollection(config['collection']) \
            .filterBounds(self.study_area_buffered) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', self.cloud_threshold))
        
        count = collection.size().getInfo()
        print(f"Found {count} images for {year}")
        
        if count == 0:
            print(f"Warning: No images found for {year}. Try adjusting settings.")
            return None, config
        
        def process_image(image):
            scaled = self.scale_landsat(image, config['collection'].split('/')[1])
            return scaled
        
        processed_collection = collection.map(process_image)
        composite = processed_collection.median().clip(self.study_area).select(config['bands'])
        
        composite = composite.set({
            'year': year,
            'system:time_start': ee.Date(start_date).millis(),
            'composite_type': 'median',
            'date_range': f"{start_date}_to_{end_date}",
            'cloud_threshold': self.cloud_threshold,
            'image_count': count
        })
        
        return composite, config

    def initialize_map(self):
        """
        Initialize geemap Map centered on study area.
        
        Returns:
            geemap.Map: Interactive map instance
        """
        centroid = self.study_area.centroid().coordinates().getInfo()
        self.map = geemap.Map(center=[centroid[1], centroid[0]], zoom=8)
        self.map.addLayer(
            self.study_area, 
            {'color': 'red', 'fillColor': '00000000', 'width': 2}, 
            'GKE Boundary'
        )
        return self.map
    
    def visualize_composites(self):
        """
        Create and visualize composites for all target years.
        
        Returns:
            geemap.Map: Map with all composite layers added
        """
        if self.map is None:
            self.initialize_map()
            
        self.composites = {}
        self.configs = {}
        vis_params = {'min': 0.0, 'max': 0.3, 'gamma': 1.4}
        
        for year in self.target_configs.keys():
            print(f"\n{'='*50}")
            print(f"Processing year: {year}")
            print(f"{'='*50}")
            
            try:
                composite, config = self.create_composite(year)
                if composite is not None:
                    self.composites[year] = composite
                    self.configs[year] = config
                    self.map.addLayer(
                        composite.select(config['bands'][0:3]),
                        vis_params,
                        f'{year} Composite',
                        False
                    )
                    print(f"✓ Successfully processed {year}")
                else:
                    print(f"✗ Failed to process {year}")
            except Exception as e:
                print(f"✗ Error processing {year}: {e}")
        
        print(f"\n{'='*50}")
        print(f"Visualization complete!")
        print(f"Processed {len(self.composites)} out of {len(self.target_configs)} years")
        
        return self.map
    
    def save_processor_state(self):
        """
        Save the processor state for export script.
        
        Returns:
            str: Path to saved state file
        """
        processor_state = {
            'shapefile_path': self.shapefile_path,
            'output_folder': self.output_folder,
            'target_configs': self.target_configs,
            'cloud_threshold': self.cloud_threshold,
            'study_area_coords': self.study_area.coordinates().getInfo() if self.study_area else None,
            'configs': self.configs if hasattr(self, 'configs') else {}
        }
        
        state_file = os.path.join(self.output_folder, 'processor_state.pkl')
        with open(state_file, 'wb') as f:
            pickle.dump(processor_state, f)
        
        print(f"\nProcessor state saved to: {state_file}")
        return state_file


# =============================================================================
# LANDSAT EXPORT CLASS
# =============================================================================

class GKELandsatExporter:
    """
    Export processed Landsat composites to Google Drive.
    
    This class handles:
        - Loading processor state from visualization workflow
        - Recreating composites for export
        - Batch export to Google Drive
        - Export progress monitoring
    
    Attributes:
        shapefile_path (str): Path to GKE boundary shapefile
        output_folder (str): Local output folder
        cloud_threshold (int): Cloud cover threshold used
        study_area: Earth Engine geometry
        composites (dict): Recreated composites
        configs (dict): Configuration metadata
    """
    
    def __init__(self, processor_state_file):
        """
        Initialize the exporter from saved processor state.
        
        Args:
            processor_state_file (str): Path to processor state pickle file
        """
        self.load_processor_state(processor_state_file)
        self.recreate_composites()
        
    def load_processor_state(self, state_file):
        """
        Load processor state from pickle file.
        
        Args:
            state_file (str): Path to state file
        """
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)

            self.shapefile_path = state['shapefile_path']
            self.output_folder = state['output_folder']
            self.cloud_threshold = state['cloud_threshold']
            self.configs = state.get('configs', {})
            self.target_configs = state.get('target_configs', {
                1984: {'start_date': '1984-05-01', 'end_date': '1984-08-31'},
                1994: {'start_date': '1994-05-01', 'end_date': '1994-08-31'},
                2004: {'start_date': '2004-05-01', 'end_date': '2004-08-31'},
                2014: {'start_date': '2014-05-01', 'end_date': '2014-08-31'},
                2024: {'start_date': '2024-05-01', 'end_date': '2024-08-31'}
            })
            
            if state['study_area_coords']:
                self.study_area = ee.Geometry.Polygon(state['study_area_coords'])
                self.study_area_buffered = self.study_area.buffer(20000)
            else:
                raise Exception("No study area coordinates found in state file")
            
            print("✓ Processor state loaded successfully!")
            print(f"Area: {self.study_area.area().divide(1000000).getInfo():.2f} km²")
            
        except Exception as e:
            print(f"Error loading processor state: {e}")
            raise
    
    def get_landsat_collection(self, year):
        """Get appropriate Landsat collection based on year."""
        if year <= 2011:
            collection = 'LANDSAT/LT05/C02/T1_L2'
            bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
        else:
            collection = 'LANDSAT/LC08/C02/T1_L2'
            bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        return {'collection': collection, 'bands': bands}
    
    def scale_landsat(self, image, landsat_type):
        """Scale Landsat surface reflectance values."""
        if landsat_type in ['LT05']:
            optical_bands = image.select(
                ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
            ).multiply(0.0000275).add(-0.2)
        else:
            optical_bands = image.select(
                ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            ).multiply(0.0000275).add(-0.2)
        return image.addBands(optical_bands, None, True)
    
    def create_composite(self, year):
        """Recreate composite for export."""
        config = self.get_landsat_collection(year)
        start_date = self.target_configs[year]['start_date']
        end_date = self.target_configs[year]['end_date']
        
        print(f"Recreating composite for {year}...")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Cloud cover threshold: {self.cloud_threshold}%")
        
        collection = ee.ImageCollection(config['collection']) \
            .filterBounds(self.study_area_buffered) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', self.cloud_threshold))
        
        count = collection.size().getInfo()
        print(f"Found {count} images for {year}")
        
        if count == 0:
            print(f"Warning: No images found for {year}.")
            return None, config
        
        def process_image(image):
            scaled = self.scale_landsat(image, config['collection'].split('/')[1])
            return scaled
        
        processed_collection = collection.map(process_image)
        composite = processed_collection.median().clip(self.study_area).select(config['bands'])
        
        composite = composite.set({
            'year': year,
            'system:time_start': ee.Date(start_date).millis(),
            'composite_type': 'median',
            'date_range': f"{start_date}_to_{end_date}",
            'cloud_threshold': self.cloud_threshold,
            'image_count': count
        })
        
        return composite, config
    
    def recreate_composites(self):
        """Recreate all composites for export."""
        print("\n" + "="*50)
        print("RECREATING COMPOSITES FOR EXPORT")
        print("="*50)
        
        self.composites = {}
        self.configs = {}
        
        for year in self.target_configs.keys():
            try:
                composite, config = self.create_composite(year)
                if composite is not None:
                    self.composites[year] = composite
                    self.configs[year] = config
                    print(f"✓ Composite recreated for {year}")
                else:
                    print(f"✗ Failed to recreate composite for {year}")
            except Exception as e:
                print(f"✗ Error recreating {year}: {e}")
        
        print(f"\nRecreated {len(self.composites)} composites for export")
    
    def export_composite(self, year, composite, config):
        """
        Export a single composite to Google Drive.
        
        Args:
            year (int): Target year
            composite: Earth Engine image
            config (dict): Configuration metadata
            
        Returns:
            ee.batch.Task: Export task
        """
        export_image = composite.select(config['bands'])
        description = f'GKE_Composite_{year}'
        
        export_params = {
            'image': export_image,
            'description': description,
            'folder': 'GKE_Landsat_Data',
            'region': self.study_area,
            'scale': 30,
            'crs': 'EPSG:32735',
            'maxPixels': 1e9,
            'fileFormat': 'GeoTIFF'
        }

        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()
        print(f"Started export for {year} Composite - Task ID: {task.id}")
        return task
    
    def export_all_composites(self):
        """
        Export all processed composites.
        
        Returns:
            dict: Export tasks keyed by year
        """
        if not hasattr(self, 'composites') or not self.composites:
            print("No composites available for export.")
            return {}
        
        export_tasks = {}
        print(f"\n{'='*50}")
        print(f"STARTING EXPORTS TO GOOGLE DRIVE")
        print(f"{'='*50}")
        
        for year in self.composites.keys():
            try:
                composite = self.composites[year]
                config = self.configs[year]
                task = self.export_composite(year, composite, config)
                export_tasks[f"{year}_composite"] = task
                print(f"✓ Export started for {year}")
            except Exception as e:
                print(f"✗ Export failed for {year}: {e}")
        
        print(f"\n{'='*50}")
        print(f"EXPORT SUMMARY")
        print(f"Total exports started: {len(export_tasks)}")
        print(f"Google Drive folder: 'GKE_Landsat_Data'")
        
        return export_tasks
    
    def monitor_exports(self, export_tasks):
        """
        Monitor export progress with periodic status updates.
        
        Args:
            export_tasks (dict): Export tasks to monitor
        """
        print(f"\n{'='*50}")
        print(f"MONITORING EXPORTS")
        print(f"{'='*50}")
        
        while True:
            completed = 0
            failed = 0
            running = 0
            
            print(f"\nStatus check at {datetime.now().strftime('%H:%M:%S')}:")
            
            for key, task in export_tasks.items():
                status = task.status()
                state = status['state']
                
                if state == 'COMPLETED':
                    completed += 1
                    print(f"  ✓ {key}: COMPLETED")
                elif state == 'FAILED':
                    failed += 1
                    error_msg = status.get('error_message', 'Unknown error')
                    print(f"  ✗ {key}: FAILED - {error_msg}")
                elif state in ['RUNNING', 'READY']:
                    running += 1
                    progress = status.get('progress', 0)
                    print(f"  ⏳ {key}: {state} ({progress:.1f}%)")
                else:
                    print(f"  ? {key}: {state}")
            
            print(f"\nSummary: {completed} completed, {running} running, {failed} failed")
            
            if completed + failed == len(export_tasks):
                print(f"\n{'='*50}")
                print(f"ALL EXPORTS COMPLETED!")
                print(f"✓ {completed} successful")
                print(f"✗ {failed} failed")
                print(f"Check your Google Drive 'GKE_Landsat_Data' folder")
                break
            
            time.sleep(30)


# =============================================================================
# MAIN WORKFLOW FUNCTIONS
# =============================================================================

def main_visualization_workflow(shapefile_path, output_folder, cloud_threshold=20):
    """
    Main workflow for GKE Landsat visualization.
    
    Args:
        shapefile_path (str): Path to study area shapefile
        output_folder (str): Output directory
        cloud_threshold (int): Maximum cloud cover percentage
        
    Returns:
        tuple: (processor instance, map widget)
    """
    print(f"""
    {'='*70}
    GKE LANDSAT VISUALIZATION
    {'='*70}
    Research: Habitat Loss Analysis (1984-2024)
    Study Area: {shapefile_path}
    Output Folder: {output_folder}
    Cloud Threshold: {cloud_threshold}%
    {'='*70}
    """)
    
    os.makedirs(output_folder, exist_ok=True)
    
    processor = GKELandsatVisualization(shapefile_path, output_folder, cloud_threshold)
    
    print("STEP 1: Loading study area...")
    if not processor.load_study_area():
        print("Failed to load study area. Exiting.")
        return None
    
    print("\nSTEP 2: Processing and visualizing composites...")
    map_widget = processor.visualize_composites()
    
    print("\nSTEP 3: Saving processor state...")
    state_file = processor.save_processor_state()
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    
    return processor, map_widget


def main_export_workflow(output_folder, monitor_progress=True):
    """
    Main workflow for exporting GKE Landsat data.
    
    Args:
        output_folder (str): Output directory containing processor state
        monitor_progress (bool): Whether to monitor export progress
        
    Returns:
        dict: Export tasks
    """
    print(f"""
    {'='*70}
    GKE LANDSAT EXPORT
    {'='*70}
    Research: Habitat Loss Analysis (1984-2024)
    Output Folder: {output_folder}
    Target: Google Drive 'GKE_Landsat_Data' folder
    {'='*70}
    """)

    state_file = os.path.join(output_folder, 'processor_state.pkl')
    if not os.path.exists(state_file):
        print(f"ERROR: Processor state file not found at {state_file}")
        return None
    
    try:
        print("STEP 1: Loading processor state...")
        exporter = GKELandsatExporter(state_file)

        print("\nSTEP 2: Starting exports...")
        export_tasks = exporter.export_all_composites()
        
        if not export_tasks:
            print("No export tasks created. Exiting.")
            return None
        
        if monitor_progress:
            print("\nSTEP 3: Monitoring export progress...")
            exporter.monitor_exports(export_tasks)
        else:
            print("\nSTEP 3: Export monitoring skipped.")
            print("\nTask IDs for manual monitoring:")
            for key, task in export_tasks.items():
                print(f"  {key}: {task.id}")
        
        return export_tasks
        
    except Exception as e:
        print(f"Export workflow failed: {e}")
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize Earth Engine
    try:
        ee.Initialize()
        print("Earth Engine initialized successfully!")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        print("Please authenticate with: ee.Authenticate()")
        exit(1)
    
    # Configuration - update these paths
    SHAPEFILE_PATH = r"D:/Publication/Simahala/Simalaha40km.shp"
    OUTPUT_FOLDER = r"D:/Publication/Simahala/outputs"
    CLOUD_THRESHOLD = 5
    
    if not os.path.exists(SHAPEFILE_PATH):
        print(f"ERROR: Shapefile not found at {SHAPEFILE_PATH}")
        exit(1)
    
    # Run visualization workflow
    processor, map_widget = main_visualization_workflow(
        SHAPEFILE_PATH, 
        OUTPUT_FOLDER, 
        cloud_threshold=CLOUD_THRESHOLD
    )
    
    if processor is not None and map_widget is not None:
        print(f"\n{'='*50}")
        print("MAP READY FOR DISPLAY")
        print(f"{'='*50}")
        print("In Jupyter notebook, display with: map_widget")
