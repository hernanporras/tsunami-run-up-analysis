# 🌊 HySEA Analysis and Visualization Toolbox

This repository contains a collection of Python scripts developed to support **tsunami modeling workflows using Tsunami-HySEA**, with emphasis on **automation, pre-processing, post-processing, and machine learning analysis**.
It integrates utilities for creating input files (parfiles), handling NetCDF/GeoTIFF/GRD data, generating GIS-ready outputs, producing visualizations (maps, GIFs, profiles), and applying **Random Forest models** to evaluate Manning’s friction coefficient influence on run-up and inundation.

---

## 📂 Repository Structure

### 🔧 Pre-processing and Model Setup

* **`Parfilero.py`** – Generator for HySEA parfiles with customizable source and simulation parameters.
* **`creardor_parfiles_and_getbalancing_hysea.py`** – Automates parfile creation and runs `get_load_balancing` for load distribution.
* **`hporras_TIF_to_grd_HYSEA_Marzo_2025_final_revhp.py`** – Converts high-resolution bathymetry/topography GeoTIFFs to `.grd` format required by HySEA.
* **`hporras_Asignar_friccionTable_a_shp.py`** – Assigns Manning friction coefficients to shapefiles for friction-variable simulations.
* **`refinamiento_mallas_2025_hysea_enero.py`** – Tools for refining and preparing nested HySEA grids.

---

### 📊 Post-processing and Visualization

* **`giff_from_nc_v4_L4.py` / `giff_from_nc_v5_L4.py`** – Generate time-evolving GIFs of tsunami propagation from NetCDF results (level L4).
* **`plot_area_curves_and_runup_profiles.py`** – Produces inundation area curves and coastal run-up profiles from HySEA outputs.
* **`export_inundation_contours_gis.py`** – Extracts inundation contours from HySEA results and exports them as GIS-ready layers.
* **`python corregir_netCDF.py`** – Corrects inconsistencies or missing variables in HySEA NetCDF outputs.
* **`inspeccionar_que_esten_todas_las_variables.py`** – Sanity check: ensures required HySEA variables are present in output NetCDF files.

---

### 📈 Machine Learning and Sensitivity Analysis

* **`manning_influence_summary_rf.py`** – Random Forest analysis of Manning coefficient influence on run-up and inundation.
* **`manning_influence_summary_rf_v2.py`** – Extended version with improved feature selection and visualizations.
* **`run-up_random_forest_analysis.py`** – Pixel-level Random Forest analysis for run-up, inundation, and momentum flux sensitivity.

---

## 🚀 Usage

1. Prepare your bathymetry and nested grids (GeoTIFF → `.grd`).
2. Generate **parfiles** with `Parfilero.py` or `creardor_parfiles_and_getbalancing_hysea.py`.
3. Run **Tsunami-HySEA** with the prepared inputs.
4. Use post-processing scripts (`plot_area_curves_and_runup_profiles.py`, `giff_from_nc_v*.py`, `export_inundation_contours_gis.py`) to analyze results.
5. Apply **Random Forest scripts** to evaluate sensitivity of Manning friction or other parameters.

---

## 🛠 Requirements

* Python 3.9+
* Tsunami-HySEA v4.3+ (GPU-enabled)
* Python libraries:

  ```bash
  pip install numpy matplotlib xarray geopandas scikit-learn obspy requests beautifulsoup4
  ```

---

## 📊 Outputs

Depending on the script, outputs may include:

* **HySEA parfiles** (`.txt`) and load balancing files (`.bin`).
* **NetCDF corrections** with consistent variables.
* **Figures**: run-up profiles, area–friction curves, Random Forest variable importance.
* **GIS layers**: inundation extent contours (shapefiles / GeoJSON).
* **GIFs**: time-evolution of tsunami surface elevation.

---

## 📜 License

**MIT License**
© 2025 Hernán Porras Espinoza

This repository is open-source for scientific and educational purposes. Attribution is appreciated in academic work.
