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

# ======================= CONFIG =======================
DIR_NC    = r"E:\INVESTIGACION\PUERTO RICO\fuentes_hernan\simulaciones\Test_FV\BoF-W"
DIR_FRIC  = r"E:\INVESTIGACION\PUERTO RICO\fuentes_hernan\batimetrias\Mallas_W_L4_e"
DIR_SLOPE = r"E:\INVESTIGACION\PUERTO RICO\fuentes_hernan\pendientes_mallas_W"
DIR_BATHY = r"E:\INVESTIGACION\PUERTO RICO\fuentes_hernan\batimetrias\mallas_inun"

OUTPUT_DIR = os.path.join(DIR_NC, "graficos_manning_2025v12")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUPS = [f"W0{i}" for i in range(1, 7)]
COLORS = plt.cm.viridis(np.linspace(0, 1, 10))
EC_FRICTION = 0.03               # fricción para 'ec'
FLOOD_TH = 0.01                  # umbral de inundación (m)

# ================== HELPERS ==================
def ec_style(e_tag):
    """Color/linewidth para ec vs resto."""
    return ("black", 1.8) if e_tag == "ec" else (None, 1.2)

def arr2d_for(sim, var):
    if var == "height":     return sim["height_2d"]
    if var == "momentum":   return sim["momentum_2d"]
    if var == "energy":     return sim["energy_2d"]
    if var == "friction":   return sim["friction_2d"]
    if var == "slope":      return sim["slope_2d"]
    if var == "bathymetry": return sim["bathymetry_2d"]
    raise KeyError(var)

def build_land_flood_mask(sim, flood_th=FLOOD_TH):
    """Máscara 2D de tierra-inundada: height>th y bathy>=0 (ambos finitos)."""
    hg = sim["height_2d"]
    b2 = sim["bathymetry_2d"]
    return np.isfinite(hg) & np.isfinite(b2) & (b2 >= 0) & (hg > flood_th)

def arr_masked_by_land_flood(sim, var, flood_th=FLOOD_TH):
    """Devuelve 1D con valores de la variable SOLO en tierra-inundada."""
    a2d  = arr2d_for(sim, var)
    mask = build_land_flood_mask(sim, flood_th)
    vals = a2d[mask]
    return vals[np.isfinite(vals)]

# ================== RECOLECCIÓN DE DATOS ==================
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

    if not (os.path.exists(slope_file) and os.path.exists(bathy_file)):
        print(f"  >> Falta slope o bathy para {group}, se omite el grupo.")
        continue

    # slope 2D (solo por si la graficas)
    with rasterio.open(slope_file) as src:
        full_slope = src.read(1)

    # bathymetry 2D
    with xr.open_dataset(bathy_file) as ds_bathy:
        full_bathy_2d = ds_bathy["z"].values

    for nc_file in nc_files:
        base_e = os.path.basename(nc_file).split("_FV_")[-1].replace(".nc", "")

        ds = xr.open_dataset(nc_file)
        height   = ds["max_height"].values  # 2D
        momentum = ds["max_mom_flux"].values  # 2D
        lon      = ds["lon"].values
        lat      = ds["lat"].values

        # Fricción: ec=0.03 donde height es finito; e1..e10 desde GRD
        if base_e == "ec":
            fric = np.where(np.isfinite(height), EC_FRICTION, np.nan).astype(float)
        else:
            fric_pattern = os.path.join(DIR_FRIC, f"*{group}*{base_e}*.grd")
            fric_files   = glob.glob(fric_pattern)
            if not fric_files:
                print(f"  >> No friction GRD para {group} {base_e}, se omite.")
                ds.close()
                continue
            fric = xr.open_dataset(fric_files[0])["z"].values

        # Área de píxel desde lon/lat (igual que hacías)
        avg_lat = float(np.nanmean(lat))
        dx_km = abs(lon[1] - lon[0]) * 111.32 * np.cos(np.deg2rad(avg_lat))
        dy_km = abs(lat[1] - lat[0]) * 110.57
        pixel_area_km2 = dx_km * dy_km

        # Guarda TODO en 2D (sin flatten, sin valid)
        energy = momentum**2

        # Perfil run-up (máx por columna)
        profile = np.array([
            np.nanmax(col) if np.any(~np.isnan(col)) else 0.0
            for col in height.T
        ])

        sim_dict = {
            "e":             base_e,
            "height_2d":     height,
            "momentum_2d":   momentum,
            "energy_2d":     energy,
            "friction_2d":   fric,
            "slope_2d":      full_slope,
            "bathymetry_2d": full_bathy_2d,
            "height_grid":   height,    # alias
            "lon":           lon,
            "profile":       profile,
            "pixel_area":    pixel_area_km2
        }
        data[group].append(sim_dict)

        # Resumen con tierra-inundada
        lf_mask = build_land_flood_mask(sim_dict, FLOOD_TH)
        flooded_pixels_land = int(lf_mask.sum())
        flooded_area_km2 = flooded_pixels_land * pixel_area_km2

        # Stats de resumen (máximos en tierra-inundada si existen; si no, global finito)
        def max_masked(arr2d, mask):
            vals = arr2d[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                vals = arr2d[np.isfinite(arr2d)]
            return float(np.nanmax(vals)) if vals.size > 0 else float("nan")

        summary.append({
            "group":            group,
            "simulation":       base_e,
            "friction_mean":    float(np.nanmean(fric[lf_mask])) if np.any(lf_mask) else float(np.nanmean(fric)),
            "height_max":       max_masked(height, lf_mask),
            "momentum_max":     max_masked(momentum, lf_mask),
            "total_energy":     float(np.nansum(energy[lf_mask])) if np.any(lf_mask) else float(np.nansum(energy)),
            "flooded_pixels":   flooded_pixels_land,
            "flooded_area_km2": float(flooded_area_km2)
        })

        ds.close()

    print(f"{group} has {len(data[group])} simulations")

# ================== CSV RESUMEN ==================
df_summary = pd.DataFrame(summary)
order = [f"e{i}" for i in range(1, 11)] + ["ec"]
if not df_summary.empty:
    df_summary["simulation"] = pd.Categorical(df_summary["simulation"], categories=order, ordered=True)
try:
    out_csv = os.path.join(OUTPUT_DIR, "resumen_simulaciones.csv")
    df_summary.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
except PermissionError:
    print("Could not save resumen_simulaciones.csv (¿abierto en Excel?)")

# ================== SUBPLOTS ÁREA ACUMULADA ==================
variables = ["height", "momentum", "energy", "friction", "slope", "bathymetry", "h"]

for var in variables:
    for scale in ["linear", "log"]:
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        axs = axs.flatten()

        for i, group in enumerate(GROUPS):
            ax = axs[i]
            sims = data[group]
            if not sims:
                ax.set_visible(False)
                continue

            if var == "h":
                h_vals = np.logspace(-2, 1, 100) if scale == "log" else np.linspace(0, 10, 100)
                for sim in sims:
                    hg = sim["height_grid"]
                    if hg.size == 0:
                        continue
                    land_mask = build_land_flood_mask(sim)  # tierra-inundada
                    area_h = [np.sum((hg > h) & land_mask) * sim["pixel_area"] for h in h_vals]
                    color, lw = ec_style(sim["e"])
                    ax.plot(h_vals, area_h, label=sim["e"], color=color, linewidth=lw)
                ax.set_xlabel("Flow height h (m)")
                ax.set_ylabel("Cumulative Flooded Area (> h) [km²]")

            else:
                for sim in sims:
                    arr = arr_masked_by_land_flood(sim, var)  # <-- tierra-inundada SIEMPRE
                    if arr.size == 0:
                        continue
                    max_val = float(np.nanmax(arr))
                    if scale == "log":
                        xmin = 5e-3 if var == "friction" else 1e-2
                        xmax = max(max_val, xmin * 10)
                        x_vals = np.logspace(np.log10(xmin), np.log10(xmax), 100)
                    else:
                        x_vals = np.linspace(0, max(max_val, 1.0), 100)

                    cum_area = [np.sum(arr > x) * sim["pixel_area"] for x in x_vals]
                    color, lw = ec_style(sim["e"])
                    ax.plot(x_vals, cum_area, label=sim["e"], color=color, linewidth=lw)

                ax.set_xlabel("Friction (Manning n)" if var == "friction" else var.capitalize())
                ax.set_ylabel("Cumulative Flooded Area (> x) [km²]")

            if scale == "log":
                ax.set_xscale("log")
            ax.set_title(group)
            ax.grid(True)
            ax.legend(fontsize=9)

        plt.suptitle(f"Cumulative Flooded Area vs {var.capitalize()} ({scale})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out = os.path.join(OUTPUT_DIR, f"Subplots_area_acum_vs_{var}_{scale}.png")
        plt.savefig(out, dpi=300)
        print("Saved:", out)
        plt.close()

# ================== PERFILES RUN-UP ==================
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
axs = axs.flatten()
for i, group in enumerate(GROUPS):
    ax = axs[i]
    sims = data[group]
    if not sims:
        ax.set_visible(False)
        continue
    for j, sim in enumerate(sims):
        color, lw = ec_style(sim["e"])
        if color is None:
            color = COLORS[j % len(COLORS)]
        ax.plot(sim["lon"], sim["profile"], label=sim["e"], color=color, linewidth=lw)
    ax.set_title(group)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Max tsunami height (m)")
    ax.grid(True)
    ax.legend(fontsize=9)
plt.suptitle("Coastal Run-Up Profiles by Group", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out = os.path.join(OUTPUT_DIR, "Subplots_perfil_runup.png")
plt.savefig(out, dpi=300)
print("Saved:", out)
plt.close()

# ================== RANDOM FOREST ==================
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
axs = axs.flatten()
if df_summary.empty:
    print("[WARN] df_summary vacío. Omitiendo RF.")
else:
    for i, group in enumerate(GROUPS):
        ax = axs[i]
        df_g = df_summary[df_summary.group == group].sort_values("simulation")
        if df_g.empty:
            ax.set_visible(False)
            continue
        X = df_g[["friction_mean", "height_max", "momentum_max", "total_energy"]].values
        y = df_g["flooded_area_km2"].values
        model = RandomForestRegressor(n_estimators=200, random_state=42)
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
    out = os.path.join(OUTPUT_DIR, "Subplots_curvas_prediccion_rf.png")
    plt.savefig(out, dpi=300)
    print("Saved:", out)
    plt.close()

# ================== DIFERENCIA: HEIGHT vs BATHY ==================
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
axs = axs.flatten()
for i, group in enumerate(GROUPS):
    ax = axs[i]
    sims = data[group]
    if not sims:
        ax.set_visible(False)
        continue
    for sim in sims:
        h2d = sim["height_2d"]
        b2d = sim["bathymetry_2d"]
        land = np.isfinite(b2d) & (b2d >= 0)
        if not np.any(land):
            continue
        x_vals = np.linspace(0, 10, 100)
        area_h = [np.sum((h2d > x) & land) * sim["pixel_area"] for x in x_vals]
        area_b = [np.sum((b2d > x) & land) * sim["pixel_area"] for x in x_vals]
        diff = np.array(area_h) - np.array(area_b)
        color, lw = ec_style(sim["e"])
        ax.plot(x_vals, diff, label=sim["e"], color=color, linewidth=lw)

    ax.set_title(group)
    ax.set_xlabel("Threshold (m)")
    ax.set_ylabel("Flooded Area Difference [km²]")
    ax.grid(True)
    ax.legend(fontsize=9)

plt.suptitle("Flooded Area Difference: Run-Up Height vs Bathymetry", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out = os.path.join(OUTPUT_DIR, "Difference_area_height_vs_bathymetry.png")
plt.savefig(out, dpi=300)
print("Saved:", out)
plt.close()

print(" Listo. Gráficos en:", OUTPUT_DIR)

