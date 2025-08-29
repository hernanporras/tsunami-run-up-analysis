import os
import glob
import numpy as np
import xarray as xr
import rasterio
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ====================================================
# CONFIGURATION: DIRECTORIES AND GROUPS
# ====================================================
DIR_NC = r"C:\Users\Hernan\OneDrive - University of Puerto Rico\Desktop\TEST_BoF-W_L4"
DIR_FRIC = r"C:\Users\Hernan\OneDrive - University of Puerto Rico\Documents\Tsunamis_tesis_II_part\batimetrias\Mallas_W_L4_e"
DIR_SLOPE = r"C:\Users\Hernan\OneDrive - University of Puerto Rico\Documents\Tsunamis_tesis_II_part\batimetrias\pendientes_mallas_W"
DIR_BATHY = r"C:\Users\Hernan\OneDrive - University of Puerto Rico\Documents\Tsunamis_tesis_II_part\batimetrias\mallas_inun"
OUTPUT_DIR = os.path.join(DIR_NC, "graficos_manning_2025v8")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GROUPS = [f"W0{i}" for i in range(1, 7)]

# ====================================================
# PIXEL-LEVEL DATA EXTRACTION
# ====================================================
pixel_records = []

for group in GROUPS:
    nc_files = sorted(glob.glob(os.path.join(DIR_NC, f"BoF-W_{group}_L4_7_5m_FV_e*.nc")))
    slope_file = os.path.join(DIR_SLOPE, f"L4_test_{group}_7_5m_nested.tif")
    bathy_file = os.path.join(DIR_BATHY, f"L4_test_{group}_7_5m_nested.grd")
    
    if not (os.path.exists(slope_file) and os.path.exists(bathy_file)):
        continue

    with rasterio.open(slope_file) as src:
        slope_full = src.read(1)
    with xr.open_dataset(bathy_file) as ds_bathy:
        bathy_full = ds_bathy['z'].values

    for nc_file in nc_files:
        base_e = os.path.basename(nc_file).split("_FV_")[-1].replace(".nc", "")
        fric_pattern = os.path.join(DIR_FRIC, f"*{group}*{base_e}*.grd")
        fric_files = glob.glob(fric_pattern)
        if not fric_files:
            continue
        fric_file = fric_files[0]

        with xr.open_dataset(fric_file) as fric_ds:
            fric = fric_ds['z'].values
        with xr.open_dataset(nc_file) as ds:
            height = ds["max_height"].values
            momentum = ds["max_mom_flux"].values
            lon = ds["lon"].values
            lat = ds["lat"].values

        height_flat = height.flatten()
        momentum_flat = momentum.flatten()
        fric_flat = fric.flatten()
        slope_flat = slope_full.flatten()
        bathy_flat = bathy_full.flatten()

        valid = (~np.isnan(height_flat) & ~np.isnan(momentum_flat) &
                 ~np.isnan(fric_flat) & ~np.isnan(slope_flat) & ~np.isnan(bathy_flat))
        if np.sum(valid) == 0:
            continue

        avg_lat = np.mean(lat)
        dx_km = abs(lon[1] - lon[0]) * 111.32 * np.cos(np.deg2rad(avg_lat))
        dy_km = abs(lat[1] - lat[0]) * 110.57
        pixel_area_km2 = dx_km * dy_km
        energy = momentum_flat[valid]**2

        n_valid = np.sum(valid)
        data_dict = {
            'group': np.full(n_valid, group),
            'friction': fric_flat[valid],
            'slope': slope_flat[valid],
            'bathymetry': bathy_flat[valid],
            'height': height_flat[valid],
            'momentum': momentum_flat[valid],
            'energy': energy,
            'pixel_area': np.full(n_valid, pixel_area_km2)
        }
        df_temp = pd.DataFrame(data_dict)
        pixel_records.append(df_temp)

# Combine all pixel records into a single DataFrame
df_pixels = pd.concat(pixel_records, ignore_index=True)
print(f"Total extracted pixels: {df_pixels.shape[0]}")

# ====================================================
# RANDOM FOREST ANALYSIS PER AREA - 4x6 SUBPLOT GRID
# ====================================================
predictor_cols = ['friction', 'slope', 'bathymetry']
targets = ['height', 'momentum']
n_areas = len(GROUPS)

fig, axs = plt.subplots(n_areas, 4, figsize=(24, 30))  # 6 rows, 4 columns

for i, group in enumerate(GROUPS):
    df_area = df_pixels[df_pixels['group'] == group]

    for j, target in enumerate(targets):
        col_offset = j * 2  # 0 for height, 2 for momentum
        if df_area.shape[0] < 100:
            axs[i, col_offset].text(0.5, 0.5, "Insufficient data", transform=axs[i, col_offset].transAxes,
                                    ha='center', va='center', fontsize=14)
            axs[i, col_offset+1].text(0.5, 0.5, "Insufficient data", transform=axs[i, col_offset+1].transAxes,
                                      ha='center', va='center', fontsize=14)
            continue

        X = df_area[predictor_cols].values
        y = df_area[target].values

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        y_pred = rf_model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        importances = rf_model.feature_importances_

        # Left subplot: Feature Importance
        axs[i, col_offset].bar(predictor_cols, importances, color='green', edgecolor='k')
        axs[i, col_offset].set_ylabel("Importance", fontsize=14)
        axs[i, col_offset].set_title(f"{group} - {target.capitalize()} - Importance", fontsize=14)
        axs[i, col_offset].set_ylim(0, 1)
        axs[i, col_offset].grid(True)
        axs[i, col_offset].tick_params(axis='both', labelsize=13)

        # Right subplot: Prediction vs Real
        axs[i, col_offset+1].scatter(y, y_pred, color='blue', alpha=0.6, edgecolor='k', s=6)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        axs[i, col_offset+1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axs[i, col_offset+1].set_xlabel("Observed", fontsize=14)
        axs[i, col_offset+1].set_ylabel("Predicted", fontsize=14)
        axs[i, col_offset+1].set_title(f"{group} - {target.capitalize()}\nMSE: {mse:.4f}, R²: {r2:.4f}", fontsize=14)
        axs[i, col_offset+1].grid(True)
        axs[i, col_offset+1].tick_params(axis='both', labelsize=13)

plt.tight_layout()
output_fig = os.path.join(OUTPUT_DIR, "random_forest_pixel_by_area_4x6.png")
plt.savefig(output_fig, dpi=300)
plt.show()

print(f"Plot saved to: {output_fig}")
