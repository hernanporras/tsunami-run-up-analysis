# -*- coding: utf-8 -*-
#Generador de GIFs (solo área W):
#- L4: W01..W06; L3: solo W; L0–L2 ignorados
#- Primero: MAX HEIGHT y MOMENTUM FLUX → 1 GIF por W (frames = e1..e10)
#- Luego: ETA → 1 GIF por archivo/escenario (W + e)
#- Fondo Esri WorldImagery + transparencia (tierra/NaN)
#"""

import os, re
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

# ================== CONFIG ==================
IN_PATH  = r"E:\INVESTIGACION\PUERTO RICO\fuentes_hernan\simulaciones\Test_FV\BoF-W"
OUT_DIR  = os.path.join(IN_PATH, "gifs_2025v2")
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 6
STRIDE = 1
CLIP_NEGATIVE_TO_NAN = True
PERC = (1, 99)  # percentiles para vmin/vmax robustos

# Basemap
USE_BASEMAP = True
DEFAULT_ZOOM = 11
BASEMAP_FAILED = False  # se resetea por GIF

# Nombres candidatos
TIME_DIM_CAND = ["time", "t", "nt", "ntime", "nTime", "Time", "frame", "frames", "step", "steps", "record", "rec"]
X_DIMS = ["x", "lon", "longitude", "i"]
Y_DIMS = ["y", "lat", "latitude", "j"]

VAR_CANDIDATES_TIME = [
    "eta", "h", "H", "height", "surface_height", "sea_surface_height",
    "water_height", "ssh", "zeta", "free_surface"
]
MAX_HEIGHT_CANDS = [
    "maximum_eta", "eta_max", "max_eta", "max_height", "Hmax", "maximum_height"
]
MOM_FLUX_CANDS = [
    "maximum_momentum_flux", "momflux_max", "max_momentum_flux", "momentum_flux"
]

# Filtro: solo W en L3/L4
W_L4_PATTERN = re.compile(r"_W0[1-6]_?", re.IGNORECASE)   # W01..W06
W_L3_ONLY    = re.compile(r"_W_", re.IGNORECASE)
L3_NOT_W     = re.compile(r"_(NE|NW|S|E)_", re.IGNORECASE)

# contextily (opcional)
try:
    import contextily as cx
    from pyproj import CRS
    HAS_CX = True
except Exception:
    HAS_CX = False
# ============================================

# ----------------- Utils -----------------
INVALID = r'[\\/:*?"<>|]+'
def sanitize(s: str) -> str:
    return re.sub(INVALID, "_", s)

def parse_level_from_name(name: str):
    m = re.search(r"_L(\d)_", name)
    return int(m.group(1)) if m else None

def norm_W_label(val):
    s = str(val).upper()
    m = re.search(r"W0?([1-6])", s)
    return f"W0{m.group(1)}" if m else s

def norm_e_label(val):
    s = str(val).lower()
    m = re.search(r"(e(?:10|[1-9]))", s)
    return m.group(1) if m else s

def parse_zone_scen_from_name(name: str):
    """Prefiere W0# si existe; si no, 'W'. Toma el ÚLTIMO e# si hay varios."""
    m_z = re.findall(r"W0[1-6]", name, flags=re.IGNORECASE)
    if m_z:
        z = norm_W_label(m_z[-1])
    else:
        m_w = re.findall(r"(?<![0-9A-Za-z])W(?![0-9A-Za-z])", name, flags=re.IGNORECASE)
        z = "W" if m_w else None
    m_e = re.findall(r"e(?:10|[1-9])", name, flags=re.IGNORECASE)
    e = norm_e_label(m_e[-1]) if m_e else None
    return z, e

def is_target_w_file(path: Path) -> bool:
    name = path.name
    lvl = parse_level_from_name(name)
    if lvl == 4:
        return W_L4_PATTERN.search(name) is not None
    if lvl == 3:
        return (W_L3_ONLY.search(name) is not None) and (L3_NOT_W.search(name) is None)
    return False

def find_dim(ds, candidates):
    for c in candidates:
        if c in ds.dims: return c
        if c in ds.coords and ds[c].sizes.get(c, 0) > 0: return c
    return None

def find_time_and_var(ds):
    """(time_dim, var_name) si hay serie temporal utilizable; si no, (None, None)."""
    tdim = find_dim(ds, TIME_DIM_CAND)
    if tdim is not None:
        for v in VAR_CANDIDATES_TIME:
            if v in ds and (tdim in ds[v].dims) and ds[v].ndim >= 2:
                return tdim, v
        any_t = [v for v in ds.data_vars if tdim in ds[v].dims and ds[v].ndim >= 2]
        if any_t:
            def score(v):
                dv, dims = ds[v], set(ds[v].dims)
                s = sum(d in dims for d in X_DIMS) + sum(d in dims for d in Y_DIMS)
                return (s, dv.sizes.get(tdim, 1))
            any_t.sort(key=score, reverse=True)
            return tdim, any_t[0]
    # 3D heurística
    for v in ds.data_vars:
        dv = ds[v]
        if dv.ndim >= 3:
            dims = list(dv.dims)
            xy = set([d for d in dims if d in X_DIMS + Y_DIMS])
            tlike = [d for d in dims if d not in xy]
            if tlike:
                tdim = max(tlike, key=lambda d: dv.sizes.get(d, 1))
                return tdim, v
    return None, None

def pick_2d_var_from_list(ds, cands):
    for v in cands:
        if v in ds and ds[v].ndim >= 2:
            return v
    for v in ds.data_vars:
        if ds[v].ndim >= 2:
            return v
    return None

def get_xy_dims(da):
    dims = list(da.dims)
    tdim = find_dim(da.to_dataset(name="tmp"), TIME_DIM_CAND)
    dims_wo_t = [d for d in dims if d != tdim]
    xdim = next((d for d in X_DIMS if d in dims_wo_t), None)
    ydim = next((d for d in Y_DIMS if d in dims_wo_t), None)
    if xdim is None or ydim is None:
        if len(dims_wo_t) >= 2:
            ydim, xdim = dims_wo_t[-2], dims_wo_t[-1]
        elif len(dims_wo_t) == 1:
            ydim, xdim = dims_wo_t[0], None
        else:
            ydim = xdim = None
    return ydim, xdim

def infer_extent(da, ydim, xdim):
    if xdim is None or ydim is None: return None
    try:
        x = da.coords[xdim].values; y = da.coords[ydim].values
        if x.ndim == 1 and y.ndim == 1:
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        elif x.ndim == 2 and y.ndim == 2:
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        else:
            return None
        vec = np.array([xmin, xmax, ymin, ymax], dtype=float)
        if not np.isfinite(vec).all(): return None
        return [xmin, xmax, ymin, ymax]
    except Exception:
        return None

def vrange_from_arrays(arrs, pct=PERC):
    vals = []
    for a in arrs:
        if a is None: continue
        v = np.asarray(a); v = v[np.isfinite(v)]
        if v.size: vals.append(v)
    if not vals: return 0.0, 1.0
    data = np.concatenate(vals)
    vmin = np.nanpercentile(data, pct[0])
    vmax = np.nanpercentile(data, pct[1])
    if np.isclose(vmin, vmax): vmax = vmin + 1e-6
    return float(vmin), float(vmax)

def vrange_from_da(da, pct=PERC):
    data = np.asarray(da.values); data = data[np.isfinite(data)]
    if data.size == 0: return 0.0, 1.0
    vmin = np.nanpercentile(data, pct[0]); vmax = np.nanpercentile(data, pct[1])
    if np.isclose(vmin, vmax): vmax = vmin + 1e-6
    return float(vmin), float(vmax)

def cmap_with_alpha(base="turbo"):
    cmap = plt.get_cmap(base).copy()
    cmap.set_bad(alpha=0.0); cmap.set_under(alpha=0.0)
    return cmap

def add_basemap(ax, extent4326, zoom=None):
    global BASEMAP_FAILED
    if BASEMAP_FAILED or not USE_BASEMAP or not HAS_CX or extent4326 is None: return
    try:
        z = DEFAULT_ZOOM if zoom is None else int(zoom)
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs=CRS.from_epsg(4326), zoom=z)
    except Exception as e:
        BASEMAP_FAILED = True
        print(f"[WARN] No se pudo cargar basemap (desactivado para este GIF): {e}")

def fig_to_rgba(fig):
    try:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        return argb[:, :, [1, 2, 3, 0]]
    except Exception:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
        return rgb

def restrict_zone_from_filename(sl, ds, zdim, src_name):
    if not zdim: return sl
    m = re.search(r"W0([1-6])", src_name, re.IGNORECASE)
    target = f"W0{m.group(1)}" if m else "W"
    for v in ds.coords[zdim].values:
        if norm_W_label(v) == norm_W_label(target):
            return sl.sel({zdim: v})
    return sl

# ------------- Render temporal (ETA) -------------
def render_eta_timeseries(da, time_dim, out_path, title=None, vmin=None, vmax=None):
    global BASEMAP_FAILED
    BASEMAP_FAILED = False
    frames = []
    T = da.sizes[time_dim]
    ydim, xdim = get_xy_dims(da)
    extent = infer_extent(da, ydim, xdim)
    cmap = cmap_with_alpha("turbo")

    for it in range(T):
        arr = da.isel({time_dim: it}).values
        if CLIP_NEGATIVE_TO_NAN and np.issubdtype(arr.dtype, np.number):
            arr = np.where(arr < 0, np.nan, arr)
        if STRIDE > 1 and arr.ndim >= 2:
            arr = arr[::STRIDE, ::STRIDE]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        fig.patch.set_alpha(0.0); ax.set_facecolor((0,0,0,0))
        if extent is not None:
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            add_basemap(ax, extent)
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest",
                           aspect="auto", extent=extent, cmap=cmap)
        else:
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest",
                           aspect="auto", cmap=cmap)

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(f"{da.name} (unid.)")
        if title: ax.set_title(f"{title}\n{time_dim}={it+1}/{T}")
        ax.set_xlabel(xdim or "x"); ax.set_ylabel(ydim or "y"); plt.tight_layout()
        frames.append(fig_to_rgba(fig)); plt.close(fig)

    imageio.mimsave(out_path, frames, fps=FPS)
    print(f"[OK] {out_path}")

# ------------- Render por W escaneando e1..e10 (2D) -------------
def render_escan_por_W(zone, e_to_path, var_candidates, out_path, title_base):
    """Un GIF por W donde cada frame es e1..e10 (2D)."""
    global BASEMAP_FAILED
    BASEMAP_FAILED = False

    arrays, extents, used_name = [], [], None
    for ei in [f"e{i}" for i in range(1,11)]:
        p = e_to_path.get(ei)
        if p is None:
            arrays.append(None); extents.append(None); continue
        ds = None
        try:
            try: ds = xr.open_dataset(p, engine="netcdf4")
            except Exception: ds = xr.open_dataset(p)
            vname = pick_2d_var_from_list(ds, var_candidates)
            if vname is None:
                arrays.append(None); extents.append(None); continue
            used_name = used_name or vname
            da = ds[vname]
            zdim = find_dim(ds, ["zone", "region", "group", "Z", "W", "w"])
            if zdim:
                target = zone
                idx = next((v for v in ds.coords[zdim].values if norm_W_label(v) == norm_W_label(target)), None)
                if idx is not None:
                    da = da.sel({zdim: idx})
            # aplana dims no espaciales tamaño 1
            while da.ndim > 2:
                for d in list(da.dims):
                    if d not in X_DIMS + Y_DIMS and da.sizes[d] == 1:
                        da = da.isel({d: 0})
                        break
                else:
                    break
            ydim, xdim = get_xy_dims(da)
            ext = infer_extent(da, ydim, xdim)
            a = da.values
            if CLIP_NEGATIVE_TO_NAN and np.issubdtype(a.dtype, np.number):
                a = np.where(a <= 0, np.nan, a)  # <=0 como tierra
            arrays.append(a); extents.append(ext)
        finally:
            try:
                if ds is not None: ds.close()
            except Exception:
                pass

    vmin, vmax = vrange_from_arrays(arrays, pct=PERC)
    extent = next((e for e in extents if e is not None), None)
    cmap = cmap_with_alpha("turbo")

    frames = []
    for idx, ei in enumerate([f"e{i}" for i in range(1,11)]):
        arr = arrays[idx]
        fig, ax = plt.subplots(figsize=(6,5), dpi=150)
        fig.patch.set_alpha(0.0); ax.set_facecolor((0,0,0,0))
        if arr is None:
            ax.text(0.5,0.5,f"{ei} (sin datos)", ha="center", va="center", transform=ax.transAxes)
            frames.append(fig_to_rgba(fig)); plt.close(fig); continue

        if STRIDE > 1 and arr.ndim >= 2:
            arr = arr[::STRIDE, ::STRIDE]

        if extent is not None:
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            add_basemap(ax, extent)
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest",
                           aspect="auto", extent=extent, cmap=cmap)
        else:
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest",
                           aspect="auto", cmap=cmap)

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label((used_name or "var") + " (unid.)")
        ax.set_title(f"{title_base} • {ei}"); plt.tight_layout()
        frames.append(fig_to_rgba(fig)); plt.close(fig)

    imageio.mimsave(out_path, frames, fps=FPS)
    print(f"[OK] {out_path}")

# ------------- ETA por dataset -------------
def process_eta_dataset(ds, src_name):
    time_dim, var_name = find_time_and_var(ds)
    if (time_dim is None) or (var_name is None):
        return
    var_name = str(var_name)
    da = ds[var_name]

    zdim = find_dim(ds, ["zone", "region", "group", "Z", "W", "w"])
    da = restrict_zone_from_filename(da, ds, zdim, src_name)

    # etiquetas desde nombre (YA CORREGIDO: W01 si existe)
    zone, esc = parse_zone_scen_from_name(src_name)
    zone = zone or "W"
    esc  = esc  or "eXX"

    vmin, vmax = vrange_from_da(da, pct=PERC)
    lvl = parse_level_from_name(src_name)
    level_tag = f"L{lvl}" if lvl is not None else "L?"
    title = f"{src_name} • {level_tag} • {zone} • {esc} • {var_name}"
    out_name = sanitize(f"{level_tag}_{zone}_{esc}_{var_name}.gif")
    out_path = os.path.join(OUT_DIR, out_name)
    render_eta_timeseries(da, time_dim, out_path, title=title, vmin=vmin, vmax=vmax)

# ------------- Agrupar L4 por W y e -------------
def group_l4_by_zone_e(paths):
    groups = {}
    for f in paths:
        name = f.name
        lvl = parse_level_from_name(name)
        if lvl != 4: continue
        # zona (CORREGIDO: detecta W01..W06 del nombre, no el primer 'W')
        z, e = parse_zone_scen_from_name(name)
        if not z or not re.match(r"W0[1-6]$", z, flags=re.IGNORECASE): continue
        if not e: continue
        groups.setdefault(z, {})[e.lower()] = f
    return groups

# --------------------- MAIN ---------------------
def main():
    in_path = Path(IN_PATH)
    if in_path.is_dir():
        candidates = list(in_path.rglob("*.nc"))
        files = [f for f in sorted(candidates) if is_target_w_file(f)]
        if not files:
            print("[INFO] No se encontraron archivos L3-W o L4-W01..W06.")
            return
    elif in_path.is_file():
        files = [in_path] if is_target_w_file(in_path) else []
        if not files:
            print(f"[INFO] {in_path.name} no es L3-W ni L4-W01..W06; se omite.")
            return
    else:
        raise FileNotFoundError(f"No existe: {IN_PATH}")

    # === PRIMERO: MAX HEIGHT y MOMENTUM FLUX por W (frames = e1..e10) ===
    l4_groups = group_l4_by_zone_e(files)
    for zone, e_to_path in l4_groups.items():
        # Max height
        out_h = os.path.join(OUT_DIR, sanitize(f"L4_{zone}_maxheight_escan.gif"))
        title_h = f"{zone} • max height (e1..e10)"
        print(f"\n=== Escaneo MAX HEIGHT: {zone} ===")
        render_escan_por_W(zone, e_to_path, MAX_HEIGHT_CANDS, out_h, title_h)

        # Momentum flux
        out_m = os.path.join(OUT_DIR, sanitize(f"L4_{zone}_momentumflux_escan.gif"))
        title_m = f"{zone} • momentum flux (e1..e10)"
        print(f"=== Escaneo MOMENTUM FLUX: {zone} ===")
        render_escan_por_W(zone, e_to_path, MOM_FLUX_CANDS, out_m, title_m)

    # === LUEGO: ETA por archivo/escenario (un GIF por e) ===
    for f in files:
        print(f"\n=== ETA temporal: {f.name} ===")
        ds = None
        try:
            try: ds = xr.open_dataset(f, engine="netcdf4")
            except Exception: ds = xr.open_dataset(f)
            process_eta_dataset(ds, f.name)
        finally:
            try:
                if ds is not None: ds.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
