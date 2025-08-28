#Hernan Porras

import os
import glob
import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

# === FIXED CONFIGURATION ===
DIR_NC = "/home/hernan/fuentes_hernan/simulaciones/Test_FV/BoF-W"
DIR_FRIC = "/home/hernan/fuentes_hernan/batimetrias/Mallas_W_L4_e"
DIR_SLOPE = "/home/hernan/analisis_simulaciones/pendientes_mallas_W"
DIR_BATHY = "/home/hernan/fuentes_hernan/batimetrias"
OUTPUT_DIR = os.path.join(DIR_NC, "graficos_manning_2025v9")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GROUPS = [f"W0{i}" for i in range(1, 7)]
COLORS = plt.cm.viridis(np.linspace(0, 1, 10))

# === DATA COLLECTION ===
data = defaultdict(list)
summary = []

for group in GROUPS:
    nc_files = sorted(glob.glob(os.path.join(DIR_NC, f"BoF-W_{group}_L4_7_5m_FV_e*.nc")))
    slope_file = os.path.join(DIR_SLOPE, f"L4_test_{group}_7_5m_nested.tif")
    bathy_file = os.path.join(DIR_BATHY, f"L4_test_{group}_7_5m_nested.grd")
    print(f"GROUP: {group}")
    print("  NC Files:", [os.path.basename(f) for f in nc_files])
    print("  SLOPE File:", os.path.basename(slope_file) if os.path.exists(slope_file) else "NOT FOUND")
    print("  BATHY File:", os.path.basename(bathy_file) if os.path.exists(bathy_file) else "NOT FOUND")

    if not os.path.exists(slope_file) or not os.path.exists(bathy_file):
        continue

    with rasterio.open(slope_file) as src:
        full_slope = src.read(1)
    with xr.open_dataset(bathy_file) as ds_bathy:
        full_bathy = ds_bathy['z'].values

    for nc_file in nc_files:
        base_e = os.path.basename(nc_file).split("_FV_")[-1].replace(".nc", "")
        fric_pattern = os.path.join(DIR_FRIC, f"*{group}*{base_e}*.grd")
        fric_files = glob.glob(fric_pattern)
        if not fric_files:
            print(f"No friction GRD file found for {base_e} in group {group}")
            continue
        fric_file = fric_files[0]

        fric_ds = xr.open_dataset(fric_file)
        fric = fric_ds['z'].values

        ds = xr.open_dataset(nc_file)
        height = ds["max_height"].values
        momentum = ds["max_mom_flux"].values
        lon = ds["lon"].values
        lat = ds["lat"].values

        height_flat = height.flatten()
        momentum_flat = momentum.flatten()
        fric_flat = fric.flatten()
        slope_flat = full_slope.flatten()
        bathy_flat = full_bathy.flatten()

        valid = ~np.isnan(height_flat) & ~np.isnan(momentum_flat) & ~np.isnan(fric_flat) & ~np.isnan(slope_flat) & ~np.isnan(bathy_flat)

        if np.sum(valid) == 0:
            print(f"No valid data for {base_e} in {group}")
            continue

        avg_lat = np.mean(lat)
        dx_km = abs(lon[1] - lon[0]) * 111.32 * np.cos(np.deg2rad(avg_lat))
        dy_km = abs(lat[1] - lat[0]) * 110.57
        pixel_area_km2 = dx_km * dy_km

        energy_flat = momentum_flat[valid]**2
        flooded_area = np.sum(height > 0.01) * pixel_area_km2

        profile = np.array([
            np.nanmax(col) if np.any(~np.isnan(col)) else 0.0
            for col in height.T
        ])

        px_data = {
            "friction": fric_flat[valid],
            "height": height_flat[valid],
            "momentum": momentum_flat[valid],
            "energy": energy_flat,
            "slope": slope_flat[valid],
            "bathymetry": bathy_flat[valid],
            "profile": profile,
            "lon": lon,
            "area": np.full(np.sum(valid), pixel_area_km2),
            "height_grid": height
        }

        data[group].append({"e": base_e, **px_data})

        summary.append({
            "group": group,
            "simulation": base_e,
            "friction_mean": np.mean(fric_flat[valid]),
            "height_max": np.max(height_flat[valid]),
            "momentum_max": np.max(momentum_flat[valid]),
            "total_energy": np.sum(energy_flat),
            "flooded_area_km2": flooded_area
        })

        ds.close()

    print(f"{group} has {len(data[group])} simulations")

# === SAVE SUMMARY CSV ===
df_summary = pd.DataFrame(summary)
try:
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "resumen_simulaciones.csv"), index=False)
except PermissionError:
    print("Could not save resumen_simulaciones.csv because it is open or in use.")

# === GROUPED PLOTS IN SUBPLOTS BY VARIABLE (linear and log) ===
variables = ["height", "momentum", "energy", "friction", "slope", "bathymetry", "h"]

for var in variables:
    for scale in ["linear", "log"]:
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        axs = axs.flatten()
        for i, group in enumerate(GROUPS):
            ax = axs[i]
            simulations = data[group]
            if not simulations:
                ax.set_visible(False)
                continue
            if var == "h":
                h_vals = np.logspace(-2, 1, 100) if scale == "log" else np.linspace(0, 10, 100)
                for sim in simulations:
                    if sim["height_grid"].size == 0:
                        continue
                    area_h = [np.sum(sim["height_grid"] > h) * sim["area"][0] for h in h_vals]
                    ax.plot(h_vals, area_h, label=sim["e"])
                ax.set_xlabel("Flow height h (m)")
                ax.set_ylabel("Cumulative Flooded Area (> h) [km²]")
            else:
                for sim in simulations:
                    if sim[var].size == 0 or np.isnan(sim[var]).all():
                        continue
                    max_val = np.nanmax(sim[var])
                    if max_val <= 0:
                        continue
                    x_vals = np.logspace(np.log10(0.01), np.log10(max_val), 100) if scale == "log" else np.linspace(0, max_val, 100)
                    cumulative_area = [np.sum(sim[var] > x) * sim["area"][0] for x in x_vals]
                    ax.plot(x_vals, cumulative_area, label=sim["e"])
                ax.set_xlabel(f"{var.capitalize()}")
                ax.set_ylabel("Cumulative Flooded Area (> x) [km²]")
            if scale == "log":
                ax.set_xscale("log")
            ax.set_title(group)
            ax.grid(True)
            ax.legend(fontsize=10)
        plt.suptitle(f"Cumulative Flooded Area vs {var.capitalize()} ({scale})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_name = f"Subplots_area_acum_vs_{var}_{scale}.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
        plt.close()

# === GROUPED RUN-UP PROFILE PLOT ===
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
axs = axs.flatten()
for i, group in enumerate(GROUPS):
    ax = axs[i]
    simulations = data[group]
    if not simulations:
        ax.set_visible(False)
        continue
    for j, sim in enumerate(simulations):
        ax.plot(sim["lon"], sim["profile"], label=sim["e"], color=COLORS[j % len(COLORS)])
    ax.set_title(group)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Max tsunami height (m)")
    ax.grid(True)
    ax.legend(fontsize=10)
plt.suptitle("Coastal Run-Up Profiles by Group", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "Subplots_perfil_runup.png"), dpi=300)
plt.close()

# === RANDOM FOREST PREDICTION CURVES IN SUBPLOTS ===
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
axs = axs.flatten()

for i, group in enumerate(GROUPS):
    ax = axs[i]
    df_g = df_summary[df_summary.group == group].sort_values("simulation")
    if df_g.empty:
        ax.set_visible(False)
        continue
    X = df_g[["friction_mean", "height_max", "momentum_max", "total_energy"]].values
    y = df_g["flooded_area_km2"].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    ax.scatter(y, y_pred, alpha=0.7, edgecolor="k", s=60)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    ax.set_title(f"{group}\nR²={r2_score(y, y_pred):.2f}, MSE={mean_squared_error(y, y_pred):.4f}")
    ax.set_xlabel("Observed [km²]")
    ax.set_ylabel("Predicted [km²]")
    ax.grid(True)

plt.suptitle("Prediction Curves (Random Forest)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "Subplots_curvas_prediccion_rf.png"), dpi=300)
plt.close()

print(" Grouped prediction curves in subplots saved successfully.")

# === DIFFERENCE: Bathymetry vs Run-Up Height ===
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
axs = axs.flatten()

for i, group in enumerate(GROUPS):
    ax = axs[i]
    simulations = data[group]
    if not simulations:
        ax.set_visible(False)
        continue

    for sim in simulations:
        if sim["height"].size == 0 or sim["bathymetry"].size == 0:
            continue

        x_vals = np.linspace(0, 10, 100)

        area_height = [np.sum(sim["height"] > x) * sim["area"][0] for x in x_vals]
        area_bathy = [np.sum(sim["bathymetry"] > x) * sim["area"][0] for x in x_vals]

        diff = np.array(area_height) - np.array(area_bathy)

        ax.plot(x_vals, diff, label=sim["e"])

    ax.set_title(group)
    ax.set_xlabel("Threshold (m)")
    ax.set_ylabel("Flooded Area Difference [km²]")
    ax.grid(True)
    ax.legend(fontsize=10)

plt.suptitle("Flooded Area Difference: Run-Up Height vs Bathymetry", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "Difference_area_height_vs_bathymetry.png"), dpi=300)
plt.close()

print(" Difference between run-up height and bathymetry curves saved successfully.")
