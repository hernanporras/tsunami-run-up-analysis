# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LightSource
from matplotlib.ticker import FixedLocator, FormatStrFormatter

import rasterio
from rasterio.transform import Affine

# shapefile (líneas de contorno)
try:
    import geopandas as gpd
    from shapely.geometry import LineString
    HAS_GPD = True
except Exception:
    HAS_GPD = False
    print("[WARN] geopandas/shapely no disponible: se omite la exportación a Shapefile.")

# ======================= CONFIG =======================
DIR_NC    = r"E:\INVESTIGACION\PUERTO RICO\fuentes_hernan\simulaciones\Test_FV\BoF-W"
DIR_BATHY = r"E:\INVESTIGACION\PUERTO RICO\fuentes_hernan\batimetrias\mallas_inun"

OUT_DIR   = os.path.join(DIR_NC, "graficos_mapas_2025v5")
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS    = [f"W0{i}" for i in range(1, 6+1)]
ALL_E     = [f"e{i}" for i in range(1, 10+1)] + ["ec"]      # e1..e10 y ec
SIM_TAGS_COLOR = ["e1", "e3", "e5", "e7", "e9", "ec"]       # mapas por grupo

FLOOD_TH  = 0.01   # m (umbral de inundación)
COAST_LVL = 0.0    # batimetría = 0 -> línea de costa

# Escalado de color
P_LO, P_HI = 2, 98
GAMMA      = 0.6

# Figura
FIGSIZE_GRID = (16, 10)
LINE_EC      = 1.0
LINE_OTHERS  = 0.5

# colores consistentes
TAG_COLORS = {f"e{i}": matplotlib.cm.tab20((i-1) % 20) for i in range(1, 11)}
TAG_COLORS["ec"] = "black"

# Transparencia en momentum/energy: máscara valores <= vmin (con pequeña tolerancia)
TOL_MIN = 1e-12

# ======================= HELPERS =======================
def load_bathy_2d(group):
    """Carga batimetría/terreno 2D desde .grd (xarray)."""
    p = os.path.join(DIR_BATHY, f"L4_test_{group}_7_5m_nested.grd")
    if not os.path.exists(p):
        print(f"[WARN] Sin bathy para {group} -> {p}")
        return None
    with xr.open_dataset(p) as ds:
        return ds["z"].values

def load_nc(group, tag):
    """Carga variables clave desde el NetCDF de una simulación."""
    p = os.path.join(DIR_NC, f"BoF-W_{group}_L4_7_5m_FV_{tag}.nc")
    if not os.path.exists(p):
        return None
    ds = xr.open_dataset(p)
    height   = ds["max_height"].values
    lon      = ds["lon"].values  # 1D
    lat      = ds["lat"].values  # 1D
    momentum = ds["max_mom_flux"].values if "max_mom_flux" in ds.variables else np.full_like(height, np.nan)
    ds.close()
    return height, momentum, lon, lat

def lonlat_to_mesh(lon, lat):
    """Convierte lon/lat 1D a mallas 2D."""
    if lon.ndim == 1 and lat.ndim == 1:
        X, Y = np.meshgrid(lon, lat)
    else:
        X, Y = lon, lat
    return X, Y

def coast_contour(ax, X, Y, B, color="k", lw=0.6, z=10):
    """Dibuja línea de costa (B=0)."""
    try:
        ax.contour(X, Y, B, levels=[COAST_LVL], colors=color, linewidths=lw, zorder=z)
    except Exception:
        pass

def inundation_contour(ax, X, Y, H, B, flood_th=FLOOD_TH, color="gray", lw=0.5, z=11):
    """Dibuja contorno de área inundada sobre tierra (H>umbral)."""
    mask_land = np.isfinite(B) & (B >= 0)
    data = np.where(mask_land, (H > flood_th).astype(float), np.nan)
    try:
        ax.contour(X, Y, data, levels=[0.5], colors=color, linewidths=lw, zorder=z)
    except Exception:
        pass

def tidy_axes(ax, show_left=False, show_right=False, show_bottom=False, show_top=False):
    """Ejes compartidos y limpios; ticks solo en bordes exteriores."""
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelleft=False, labelright=False, labelbottom=False, labeltop=False,
                   left=False, right=False, bottom=False, top=False)
    if show_left:
        ax.tick_params(labelleft=True, left=True)
    if show_right:
        ax.tick_params(labelright=True, right=True)
    if show_bottom:
        ax.tick_params(labelbottom=True, bottom=True)
    if show_top:
        ax.tick_params(labeltop=True, top=True)

def set_deg_ticks(ax, lon, lat, step=0.04, show_left=False, show_bottom=False):
    """Ticks cada 'step' grados (solo bordes exteriores)."""
    xmin, xmax = float(np.nanmin(lon)), float(np.nanmax(lon))
    ymin, ymax = float(np.nanmin(lat)), float(np.nanmax(lat))

    def _grid(vmin, vmax, s):
        start = np.floor(vmin / s) * s
        end   = np.ceil (vmax / s) * s + 1e-12
        return np.arange(start, end, s)

    if show_bottom:
        xticks = _grid(xmin, xmax, step)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelbottom=True, bottom=True)

    if show_left:
        yticks = _grid(ymin, ymax, step)
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelleft=True, left=True)

def common_vmin_vmax(arrs2d):
    """vmin/vmax comunes usando percentiles sobre arrays válidos."""
    vals = []
    for a in arrs2d:
        if a is None:
            continue
        good = a[np.isfinite(a)]
        if good.size:
            vals.append(good.ravel())
    if not vals:
        return None, None
    vals = np.concatenate(vals)
    vmin = np.percentile(vals, P_LO)
    vmax = np.percentile(vals, P_HI)
    if vmin == vmax:
        vmax = vmin + (abs(vmin) if vmin != 0 else 1.0)
    return float(vmin), float(vmax)

def draw_hillshade(ax, lon, lat, bathy2d, azdeg=315, altdeg=45, vert_exag=1.0, alpha=0.95):
    """Hillshade de batimetría (relieve sombreado) como fondo."""
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    b = np.array(bathy2d, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p2, p98 = np.nanpercentile(b, 2), np.nanpercentile(b, 98)
        if p98 > p2:
            b = (b - p2) / (p98 - p2)
    rgb = ls.shade(b, cmap=plt.get_cmap("Greys"), vert_exag=vert_exag, blend_mode='overlay')
    extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
    ax.imshow(rgb, extent=extent, origin="lower", zorder=0, alpha=alpha, interpolation="bilinear")

# ======= GeoTIFF (sin espejo) =======
def _axis_info_1d(arr):
    """Devuelve (vmin, vmax, n, ascending:bool)."""
    return float(np.nanmin(arr)), float(np.nanmax(arr)), len(arr), bool(arr[-1] > arr[0])

def save_geotiff(path, data2d, lon, lat, nodata=-9999.0, dtype='float32'):
    """
    Guarda GeoTIFF en EPSG:4326 ajustando el array para que:
    - la columna 0 sea el oeste (lon creciente → dx>0; si no, se voltea L/R),
    - la fila 0 sea el norte (lat norte arriba; si lat ascendente sur→norte, se voltea U/D),
    - transform = T(x_west, y_north) * S(dx, -dy).
    """
    xmin, xmax, nx, lon_asc = _axis_info_1d(lon)
    ymin, ymax, ny, lat_asc = _axis_info_1d(lat)

    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 1.0
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 1.0

    out = np.array(data2d, dtype=np.float32)

    # Asegurar x creciente (oeste→este)
    if not lon_asc:
        out = np.fliplr(out)

    # Asegurar fila 0 = norte
    if lat_asc:   # sur→norte -> voltear vertical
        out = np.flipud(out)
        y_north = ymax
    else:         # norte→sur
        y_north = lat[0]  # ≈ ymax

    x_west = xmin
    transform = Affine.translation(x_west, y_north) * Affine.scale(dx, -dy)

    out[~np.isfinite(out)] = nodata

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)

# ======= Contornos a Shapefile (líneas) =======
def inundation_lines_gdf(X, Y, H, B, flood_th=0.01, tag="e1"):
    if not HAS_GPD:
        return None
    # máscara tierra + booleano de inundación
    mask_land = np.isfinite(B) & (B >= 0)
    Z = np.where(mask_land, (H > flood_th).astype(float), np.nan)

    # hacer contornos en figura temporal
    fig = plt.figure(figsize=(2, 2))
    cs = plt.contour(X, Y, Z, levels=[0.5])
    geoms = []
    for col in cs.collections:
        for p in col.get_paths():
            v = p.vertices
            if len(v) >= 2:
                geoms.append(LineString(v))
    plt.close(fig)

    if not geoms:
        return gpd.GeoDataFrame(columns=["tag", "geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame({"tag":[tag]*len(geoms), "geometry":geoms}, geometry="geometry", crs="EPSG:4326")
    return gdf

def export_contours_shapefile(group, outdir=OUT_DIR):
    if not HAS_GPD:
        return
    bathy = load_bathy_2d(group)
    if bathy is None:
        print(f"[WARN] Sin bathy para {group} (shp)")
        return
    lon = lat = None
    for tag in ALL_E:
        nc = load_nc(group, tag)
        if nc is not None:
            _, _, lon, lat = nc
            break
    if lon is None:
        print(f"[WARN] Sin lon/lat para {group} (shp)")
        return

    X, Y = lonlat_to_mesh(lon, lat)
    gdfs = []
    for tag in ALL_E:
        nc = load_nc(group, tag)
        if nc is None:
            continue
        H, _, _, _ = nc
        gdf_tag = inundation_lines_gdf(X, Y, H, bathy, flood_th=FLOOD_TH, tag=tag)
        if gdf_tag is not None and len(gdf_tag) > 0:
            gdfs.append(gdf_tag)

    if not gdfs:
        print(f"[WARN] No contours for {group}")
        return

    gdf_all = gpd.pd.concat(gdfs, ignore_index=True)
    shp_out = os.path.join(outdir, f"Contours_{group}.shp")
    gdf_all.to_file(shp_out)
    print("Saved:", shp_out)

# ======================= FIGURA A: CONTORNOS (2x3) =======================
def figure_outlines_allE_2x3(outdir=OUT_DIR):
    fig, axs = plt.subplots(
        2, 3, figsize=FIGSIZE_GRID, sharex=True, sharey=True,
        gridspec_kw={'wspace':0, 'hspace':0}
    )
    axs = axs.flatten()

    meta = []  # {'ax':ax, 'lon':lon, 'lat':lat, 'r':r, 'c':c}

    for i, group in enumerate(GROUPS):
        ax = axs[i]
        bathy = load_bathy_2d(group)
        if bathy is None:
            ax.set_visible(False); continue

        lon = lat = None
        for tag in ALL_E:
            nc = load_nc(group, tag)
            if nc is not None:
                _, _, lon, lat = nc
                break
        if lon is None:
            ax.set_visible(False); continue

        X, Y = lonlat_to_mesh(lon, lat)

        lat_span = float(np.nanmax(lat) - np.nanmin(lat))
        lon_span = float(np.nanmax(lon) - np.nanmin(lon))
        ax.set_box_aspect(lat_span / lon_span)

        draw_hillshade(ax, lon, lat, bathy, alpha=0.95)
        coast_contour(ax, X, Y, bathy, color="k", lw=0.6, z=10)

        # contornos por color; ec más grueso
        for tag in ALL_E:
            nc = load_nc(group, tag)
            if nc is None:
                continue
            H, _, _, _ = nc
            color = TAG_COLORS.get(tag, "0.6")
            lw = LINE_OTHERS if tag != "ec" else LINE_EC
            inundation_contour(ax, X, Y, H, bathy, color=color, lw=lw, z=12 if tag!="ec" else 13)

        ax.text(0.01, 0.99, group, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, weight="bold", color="w",
                bbox=dict(facecolor="0", alpha=0.25, edgecolor="none", boxstyle="round,pad=0.2"))

        r, c = divmod(i, 3)
        meta.append({'ax': ax, 'lon': lon, 'lat': lat, 'r': r, 'c': c})

    nrows, ncols = 2, 3
    for m in meta:
        ax, lon, lat, r, c = m['ax'], m['lon'], m['lat'], m['r'], m['c']
        tidy_axes(ax,
                  show_left=(c==0), show_right=(c==ncols-1),
                  show_bottom=(r==nrows-1), show_top=(r==0))
        set_deg_ticks(ax, lon, lat, step=0.04,
                      show_left=(c==0), show_bottom=(r==nrows-1))

    plt.subplots_adjust(wspace=0, hspace=0)
    out = os.path.join(outdir, "FigA_contornos_todasE_hillshade.png")
    plt.savefig(out, dpi=350, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

# ======================= FIGURA B: MAPAS + CONTORNOS (por grupo) =======================
def figure_maps_per_group(group, var="height", cmap="viridis", outdir=OUT_DIR):
    """
    Por grupo: 2x3 (e1,e3,e5,e7,e9,ec):
      - Fondo: hillshade.
      - Mapa var (height/momentum/energy) con PowerNorm + percentiles (2–98).
      - Contornos de TODAS las simulaciones por color; ec negro más grueso.
      - Ejes pegados; ticks 0.04° en bordes; exporta GeoTIFF por panel.
    """
    bathy = load_bathy_2d(group)
    if bathy is None:
        print(f"[WARN] Sin bathy para {group}")
        return

    H_list, M_list, lon, lat = [], [], None, None
    for tag in SIM_TAGS_COLOR:
        nc = load_nc(group, tag)
        if nc is None:
            H_list.append(None); M_list.append(None); continue
        H, M, lo, la = nc
        if lon is None:
            lon, lat = lo, la
        H_list.append(H); M_list.append(M)

    if lon is None:
        print(f"[WARN] Sin datos para {group}")
        return

    X, Y = lonlat_to_mesh(lon, lat)
    mask_land = np.isfinite(bathy) & (bathy >= 0)

    stacks = []
    for H, M in zip(H_list, M_list):
        if H is None:
            continue
        if var == "height":
            A = H
        elif var == "momentum":
            A = M
        else:  # energy ~ momentum^2
            A = M**2
        stacks.append(np.where(mask_land, A, np.nan))

    vmin, vmax = common_vmin_vmax(stacks)
    if vmin is None:
        print(f"[WARN] No data for {group}, {var}")
        return
    norm = PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(
        2, 3, figsize=FIGSIZE_GRID, sharex=True, sharey=True,
        gridspec_kw={'wspace':0, 'hspace':0}
    )
    axs = axs.flatten()

    lat_span = float(np.nanmax(lat) - np.nanmin(lat))
    lon_span = float(np.nanmax(lon) - np.nanmin(lon))
    box_aspect = lat_span / lon_span

    base_cmap = plt.get_cmap(cmap).copy()
    base_cmap.set_bad(alpha=0.0)  # enmascarado será transparente

    im_for_cbar = None
    meta = []  # para ticks

    for i, tag in enumerate(SIM_TAGS_COLOR):
        ax = axs[i]
        nc = load_nc(group, tag)
        if nc is None:
            ax.set_visible(False); continue
        H, M, _, _ = nc

        if var == "height":
            A = H
        elif var == "momentum":
            A = M
        else:
            A = M**2

        Amap = np.where(mask_land, A, np.nan)

        # Guardar GeoTIFF (campo numérico útil sobre tierra) SIN ESPEJO
        gt_out = os.path.join(outdir, f"GT_{group}_{tag}_{var}.tif")
        save_geotiff(gt_out, Amap, lon, lat)

        # Transparencia del mínimo (solo momentum/energy)
        if var in ("momentum", "energy"):
            Aplot = np.ma.masked_where(np.isfinite(Amap) & (Amap <= (vmin + TOL_MIN)), Amap)
        else:
            Aplot = np.ma.masked_invalid(Amap)

        ax.set_box_aspect(box_aspect)
        draw_hillshade(ax, lon, lat, bathy, alpha=0.95)

        im = ax.pcolormesh(X, Y, Aplot, shading="auto", cmap=base_cmap, norm=norm, zorder=5)
        if im_for_cbar is None:
            im_for_cbar = im

        coast_contour(ax, X, Y, bathy, color="k", lw=0.6, z=12)

        # contornos de todas las simulaciones
        for t2 in ALL_E:
            nc2 = load_nc(group, t2)
            if nc2 is None:
                continue
            H2, _, _, _ = nc2
            color = TAG_COLORS.get(t2, "0.6")
            lw = LINE_OTHERS if t2 != "ec" else LINE_EC
            inundation_contour(ax, X, Y, H2, bathy, color=color, lw=lw, z=13 if t2!="ec" else 14)

        ax.text(0.01, 0.99, f"{group} – {tag}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="w",
                bbox=dict(facecolor="0", alpha=0.25, edgecolor="none", boxstyle="round,pad=0.15"))

        r, c = divmod(i, 3)
        meta.append({'ax': ax, 'lon': lon, 'lat': lat, 'r': r, 'c': c})

    # ejes exteriores + ticks 0.04° + sin espacios
    nrows, ncols = 2, 3
    for m in meta:
        ax, lo, la, r, c = m['ax'], m['lon'], m['lat'], m['r'], m['c']
        tidy_axes(ax,
                  show_left=(c==0), show_right=(c==ncols-1),
                  show_bottom=(r==nrows-1), show_top=(r==0))
        set_deg_ticks(ax, lo, la, step=0.04,
                      show_left=(c==0), show_bottom=(r==nrows-1))
    plt.subplots_adjust(wspace=0, hspace=0)

    # colorbar compacta
    if im_for_cbar is not None:
        cb = fig.colorbar(im_for_cbar, ax=axs, fraction=0.025, pad=0.01)
        label = {"height": "Max height (m)", "momentum": "Max momentum", "energy": "Energy (momentum²)"}[var]
        cb.set_label(label)

    out = os.path.join(outdir, f"FigB_{group}_{var}_hillshade.png")
    plt.savefig(out, dpi=350, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

# ======================= RUN =======================
if __name__ == "__main__":
    # A) Contornos de TODAS las simulaciones, 2x3, sin espacios, con hillshade + ticks 0.04°
    figure_outlines_allE_2x3(outdir=OUT_DIR)

    # B) Por grupo: height/momentum/energy, con transparencia en mínimos y GeoTIFF sin espejo
    for g in GROUPS:
        figure_maps_per_group(g, var="height",   cmap="viridis", outdir=OUT_DIR)
        figure_maps_per_group(g, var="momentum", cmap="jet",     outdir=OUT_DIR)
        figure_maps_per_group(g, var="energy",   cmap="jet",     outdir=OUT_DIR)

        # C) Shapefile de contornos por grupo
        export_contours_shapefile(g, outdir=OUT_DIR)
