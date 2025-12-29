# Habitat Loss Analysis in the Greater Kafue Ecosystem

**A comprehensive analytical framework for assessing habitat transformation, identifying drivers, and projecting future scenarios in Zambia's largest conservation landscape.**

---

## Citation

If you use this code, please cite:

```
Mulenga, G. (2024). Habitat Loss Assessment in the Greater Kafue Ecosystem: 
A Multi-temporal Analysis Using Remote Sensing and Machine Learning (1984-2024). 
MSc Thesis, Copperbelt University, Zambia.
```

---

## Overview

This repository contains Python scripts for analyzing habitat loss patterns in the Greater Kafue Ecosystem (GKE), a 66,000 km² conservation landscape in Zambia comprising Kafue National Park and nine surrounding Game Management Areas.

The analysis framework addresses three research objectives:

1. **Objective A**: Spatiotemporal analysis of habitat loss patterns (1984-2024)
2. **Objective B**: Identification and ranking of anthropogenic drivers using Random Forest
3. **Objective C**: Future scenario projections using CA-Markov modeling (2024-2050)

---

## Repository Structure

```
├── 01_gke_landsat_processing_visualization.py   # Landsat data processing
├── 02_gke_objective_a_analysis.py               # Spatiotemporal habitat analysis
├── 03_data_acquisition_gee.py                   # GEE driver variable acquisition
├── 04_sample_generation.py                      # Stratified sampling & VIF analysis
├── 05_random_forest_analysis.py                 # Driver importance & thresholds
├── 06_susceptibility_projections.py             # Susceptibility mapping
├── 07_transition_matrix_analysis.py             # Transition probability analysis
├── 08_ca_markov_scenarios.py                    # Multi-scenario projections
├── 09_susceptibility_by_zone.py                 # Zone-based susceptibility analysis
├── 10_gma_visualization.py                      # GMA land cover change visualizations
├── 11_knp_gradient_visualization.py             # KNP boundary gradient analysis
├── 12_net_change_visualization.py               # Decadal net change visualizations
└── README.md                                    # This file
```

---

## Scripts Description

### 01_gke_landsat_processing_visualization.py

Processes multi-temporal Landsat imagery (1984-2024) using Google Earth Engine.

**Key Features:**
- Cloud masking and atmospheric correction
- Spectral index calculation (NDVI, NDWI, NDBI)
- Composite generation for classification epochs
- Export to Google Drive

**Dependencies:** `earthengine-api`, `geemap`, `numpy`

---

### 02_gke_objective_a_analysis.py

Comprehensive spatiotemporal analysis of habitat transformation.

**Key Features:**
- Area statistics calculation by land cover class
- Transition matrix generation
- Distance gradient analysis from protected area boundaries
- Net change and annual rate calculations
- Publication-quality visualizations

**Dependencies:** `rasterio`, `geopandas`, `numpy`, `pandas`, `matplotlib`, `seaborn`

---

### 03_data_acquisition_gee.py

Acquires and preprocesses driver variables using Google Earth Engine.

**Variables Acquired:**
- **Proximity**: Distance to roads, settlements, rivers, KNP
- **Socio-economic**: Population density, population change, cultivation percentage
- **Conservation**: Protection status
- **Topographic**: Elevation, slope, aspect, TWI
- **Climatic**: Mean rainfall, mean temperature

**Dependencies:** `earthengine-api`, `geemap`, `geopandas`, `numpy`

---

### 04_sample_generation.py

Generates stratified random samples for Random Forest training.

**Key Features:**
- Binary habitat loss mask creation
- Spatial stratification with minimum distance constraint
- Multi-collinearity analysis (VIF)
- Spatially-blocked train-test split

**Dependencies:** `geopandas`, `rasterio`, `numpy`, `pandas`, `sklearn`, `scipy`

---

### 05_random_forest_analysis.py

Random Forest classification for driver importance analysis.

**Key Features:**
- Recursive Feature Elimination (RFE)
- Spatial cross-validation
- Gini and permutation importance
- Critical threshold identification (Youden's J)
- Partial dependence plots

**Dependencies:** `sklearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`

---

### 06_susceptibility_projections.py

Generates habitat loss susceptibility maps.

**Key Features:**
- Probability surface generation from RF model
- Susceptibility classification
- Transition matrix calculation
- Area statistics by susceptibility class

**Dependencies:** `rasterio`, `numpy`, `pandas`, `matplotlib`, `sklearn`, `joblib`

---

### 07_transition_matrix_analysis.py

Standalone utility for transition matrix analysis.

**Key Features:**
- Multi-period transition matrix calculation
- Annualization using matrix logarithm
- Net change statistics
- Transition heatmap visualization

**Dependencies:** `rasterio`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`

---

### 08_ca_markov_scenarios.py

CA-Markov modeling for multi-scenario projections.

**Scenarios:**
- **Business-as-Usual (BAU)**: Historical rates continue
- **Enhanced Conservation**: Strengthened protection
- **Accelerated Development**: Increased agricultural expansion

**Key Features:**
- Ensemble simulations (n=10)
- Neighborhood contiguity effects
- Susceptibility integration
- Projections for 2030, 2040, 2050

**Dependencies:** `rasterio`, `numpy`, `pandas`, `matplotlib`, `scipy`, `joblib`

---

### 09_susceptibility_by_zone.py

Zone-based susceptibility analysis.

**Management Zones:**
- Kafue National Park (KNP)
- Game Management Areas (GMAs)
- Open Areas

**Key Features:**
- Zone mask generation
- Statistics by management zone
- Individual GMA ranking
- Comparative visualizations

**Dependencies:** `rasterio`, `geopandas`, `numpy`, `pandas`, `matplotlib`, `seaborn`

---

## Requirements

### Python Environment

```bash
# Create conda environment
conda create -n gke_analysis python=3.10
conda activate gke_analysis

# Core dependencies
pip install numpy pandas geopandas rasterio shapely
pip install scikit-learn scipy joblib
pip install matplotlib seaborn

# Google Earth Engine (for scripts 01, 03)
pip install earthengine-api geemap

# Optional for interactive maps
pip install folium
```

### Required Python Packages

```
numpy>=1.21.0
pandas>=1.3.0
geopandas>=0.10.0
rasterio>=1.2.0
shapely>=1.8.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
earthengine-api>=0.1.300
geemap>=0.10.0
```

---

## Data Requirements

### Input Data Structure

```
data/
├── GKE_Boundary.shp          # Study area boundary
├── nationalParks.shp         # Kafue National Park boundary
├── GMA.shp                   # Game Management Area boundaries
├── Roads.shp                 # Road network
├── GKE_1984.tif              # Classified land cover 1984
├── GKE_1994.tif              # Classified land cover 1994
├── GKE_2004.tif              # Classified land cover 2004
├── GKE_2014.tif              # Classified land cover 2014
└── GKE_2024.tif              # Classified land cover 2024
```

### Land Cover Classification Schema

| Code | Class     | Description                    |
|------|-----------|--------------------------------|
| 1    | Built-up  | Urban and developed areas      |
| 2    | Forest    | Closed and open woodland       |
| 3    | Cropland  | Agricultural areas             |
| 4    | Grassland | Natural grasslands and shrubs  |
| 5    | Bareland  | Exposed soil and rock          |
| 6    | Water     | Rivers, lakes, wetlands        |

---

## Usage

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/GiftMulenga/gke-habitat-analysis.git
   cd gke-habitat-analysis
   ```

2. **Configure paths**
   
   Edit the `Config` class in each script to point to your data directories.

3. **Run analysis**
   ```bash
   # Objective A: Spatiotemporal analysis
   python 02_gke_objective_a_analysis.py
   
   # Objective B: Driver analysis
   python 04_sample_generation.py
   python 05_random_forest_analysis.py
   
   # Objective C: Future projections
   python 08_ca_markov_scenarios.py
   ```

### Workflow Sequence

```
Objective A                    Objective B                    Objective C
-----------                    -----------                    -----------
01 → 02                        03 → 04 → 05 → 06              07 → 08
                                        ↓
                               09 (Zone Analysis)
```

---

## Key Findings

### Habitat Change (1984-2024)

- Natural habitat declined from 97.63% to 82.53% of the study area
- Agricultural expansion accounts for 83% of habitat conversion
- Annual loss rate: 0.38% of remaining natural habitat
- Critical cultivation threshold: ~9-15%

### Top Drivers (Random Forest Analysis)

1. Percentage cultivated land (32.4% importance)
2. Distance to roads (18.7%)
3. Population density (12.3%)
4. Distance to settlements (9.8%)
5. Distance to KNP boundary (8.2%)

### Scenario Projections (2050)

| Scenario              | Natural Habitat | Change from 2024 |
|-----------------------|-----------------|------------------|
| Business-as-Usual     | 71.2%           | -11.3%           |
| Enhanced Conservation | 79.8%           | -2.7%            |
| Accelerated Development | 62.4%         | -20.1%           |

---

## Outputs

Each script generates outputs in organized directories:

```
outputs/
├── figures/          # Publication-quality visualizations
├── tables/           # CSV tables for analysis results
├── maps/             # GeoTIFF raster outputs
├── samples/          # Training/test data
└── projections/      # Scenario projection rasters
```

---

## License

This code is provided for academic and research purposes. Please cite appropriately.

---

## Contact

**Gift Mulenga**  
MSc Tropical Ecology  
Copperbelt University, Zambia  

---

## Acknowledgments

- Zambia Wildlife Authority (ZAWA)
- Department of National Parks and Wildlife
- Copperbelt University, School of Natural Resources
