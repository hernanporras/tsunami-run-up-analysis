# -*- coding: utf-8 -*-
"""
GIFs lentos y robustos (solo área W):
- L4: W01..W06; L3: solo W; L0–L2 ignorados
- 1) MAX HEIGHT y MOMENTUM FLUX → 1 GIF por W (frames = e1..e10)
- 2) ETA → 1 GIF por archivo/escenario (W + e)
- Basemap Esri a alta resolución con fallback de zoom (evita reventar por disco)
- Transparencia: tierra/negativos y flux==0
- Proporciones de la malla preservadas, colorbar sin cortes
- GIFs lentos garantizados duplicando frames con Pillow
"""

import os, re
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ================== CONFIG ==================
IN_PATH  = r"E:\INVESTIGACION\PUERTO RICO\fuentes_hernan\simulaciones\Test_FV\BoF-W"
OUT_DIR  = os.path.join(IN_PATH, "gifs_2025v9")
os.makedirs(OUT_DIR, exist_ok=True)

# *** IMPORTANTE: pon aquí un disco/carpeta con MUCHO espacio libre ***
TMP_DIR = r"E:\INVESTIGACION\temp_rio"
os.makedirs(TMP_DIR, exist_ok=True)
os.environ.setdefault("RIO_TMPDIR", TMP_DIR)
os.environ.setdefault("CPL_TMPDIR", TMP_DIR)
os.environ.setdefault("GDAL_CACHEMAX", "256")  # MB

# Duraciones “conceptuales” (en segundos)
ETA_SECS_PER_FRAME    = 1.0
ESCAN_SECS_PER_FRAME  = 1.0
INTRO_BASEMAP_SECS    = 1.0
HOLD_SECS             = 1.0

# Subframes para forzar lentitud en visores mañosos
SUBFRAME_MS = 1000  # 1 s por subframe (sube a 1500–2000 si aún va rápido)

INTRO_BASEMAP         = True
INTRO_BASEMAP_FRAMES  = 1
STAMP_E_LABEL         = True
STAMP_FS              = 22
STAMP_BOX = dict(facecolor='black', alpha=0.55, edgecolor='none', boxstyle='round,pad=0.25')

# Tamaños base (altura) y tipografías
FIGSIZE   = (12.0, 10.0)   # usamos FIGSIZE[1] como alto base, ancho se calcula por proporción
DPI       = 240
TITLE_FS  = 19
LABEL_FS  = 12
TICK_FS   = 11
CB_FS     = 11
TITLE_BOX = dict(facecolor='white', alpha=0.95, edgecolor='none', boxstyle='round,pad=0.6')

# Procesamiento
STRIDE = 1
CLIP_NEGATIVE_TO_NAN = True
PERC = (1, 99)

# Transparencia específica para MOMENTUM FLUX
FLUX_ZERO_EPS = 0.0   # si quieres “limpiar” valores muy pequeños: 1e-9 ó 1e-6

# Basemap
USE_BASEMAP    = True
DEFAULT_ZOOM   = 14              # sugerido; el sistema bajará si es pesado
MIN_FALLBACK_Z = 14              # no bajar de este zoom salvo catástrofe
BASEMAP_FAILED = False

# Variables y dims
TIME_DIM_CAND = ["time", "t", "nt", "ntime", "nTime", "Time", "frame", "frames", "step", "steps", "record", "rec"]
X_DIMS = ["x", "lon", "longitude", "i"]
Y_DIMS = ["y", "lat", "latitude", "j"]

VAR_CANDIDATES_TIME = [
    "eta", "h", "H", "height", "surface_height", "sea_surface_height",
    "water_height", "ssh", "zeta", "free_surface"
]
MAX_HEIGHT_CANDS = [
    "max_height",
    "maximum_eta", "eta_max", "max_eta", "Hmax", "maximum_height"
]
MOM_FLUX_CANDS = [
    "max_mom_flux",
    "maximum_momentum_flux", "momflux_max", "max_momentum_flux", "momentum_flux"
]

# Filtro: solo W en L3/L4
W_L4_PATTERN = re.compile(r"_W0[1-6]_?", re.IGNORECASE)
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
    tdim = find_dim(ds, TIME_DIM_CAND)
    if tdim is not None:
        for v in VAR_CANDIDATES_TIME:
            if v in ds and (tdim in ds[v].dims) and ds[v].ndim >= 2:
                return tdim, v
        any_t = [v for v in ds.data_vars if tdim in ds[v].dims and ds[v].ndim >= 2]
        if any_t:
            def score(v):
                dv, dims = ds[v], set(ds[v].dims)
                s = sum(d in dims for d in X_DIMS) + sum(d in Y_DIMS)
                return (s, dv.sizes.get(tdim, 1))
            any_t.sort(key=score, reverse=True)
            return tdim, any_t[0]
    # heurística 3D
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

def pick_strict_2d_var(ds, cands):
    for v in cands:
        if v in ds and ds[v].ndim >= 2:
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
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
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

# ---------- Basemap con fallback de zoom ----------
def add_basemap_smart(ax, extent4326, zoom_hint=None, min_zoom=MIN_FALLBACK_Z):
    """
    Intenta añadir basemap empezando en zoom_hint y, si falla (espacio/memoria),
    baja el zoom hasta min_zoom. Devuelve el zoom usado o None.
    """
    global BASEMAP_FAILED
    if BASEMAP_FAILED or not USE_BASEMAP or not HAS_CX or extent4326 is None:
        return None
    try:
        import contextily as cx
        from pyproj import CRS
    except Exception:
        BASEMAP_FAILED = True
        return None

    z_start = int(zoom_hint) if zoom_hint is not None else int(DEFAULT_ZOOM)
    for z in range(z_start, int(min_zoom) - 1, -1):
        try:
            cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                           crs=CRS.from_epsg(4326), zoom=z)
            print(f"[INFO] Basemap OK con zoom {z}")
            return z
        except Exception as e:
            msg = str(e)
            print(f"[WARN] Basemap falló en zoom {z}: {msg[:180]}")
            continue

    BASEMAP_FAILED = True
    print("[WARN] Basemap desactivado en este GIF (falló todos los zooms).")
    return None

def fig_to_rgba(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        return argb[:, :, [1, 2, 3, 0]]
    except Exception:
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

# ------ Colorbar sin cortes ------
def add_colorbar(im, ax, label_text):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad="2.5%")
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label_text, fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=CB_FS)
    return cb

# ------ Proporciones ------
def _is_lonlat_extent(ext):
    if ext is None:
        return False
    xmin, xmax, ymin, ymax = ext
    return (-200 <= xmin <= 200) and (-200 <= xmax <= 200) and (-90 <= ymin <= 90) and (-90 <= ymax <= 90)

def _figsize_from_extent(extent, base_height=10.0, min_w=6.0, max_w=18.0):
    if extent is None:
        return (FIGSIZE[0], FIGSIZE[1])
    xmin, xmax, ymin, ymax = extent
    width = max(xmax - xmin, 1e-9)
    height = max(ymax - ymin, 1e-9)
    ratio = width / height
    if _is_lonlat_extent(extent):
        lat_mid = 0.5 * (ymin + ymax)
        ratio *= max(np.cos(np.deg2rad(lat_mid)), 1e-4)
    fig_w = np.clip(base_height * ratio, min_w, max_w)
    return (float(fig_w), float(base_height))

# ------ GIF lento con Pillow ------
def save_gif_ultraslow(out_path, frames_np, secs_list, subframe_ms=1000):
    assert len(frames_np) == len(secs_list)
    imgs, durations = [], []
    for fr, secs in zip(frames_np, secs_list):
        if fr.ndim == 3 and fr.shape[2] >= 3:
            im_rgb = Image.fromarray(fr[:, :, :3])
        else:
            im_rgb = Image.fromarray(fr)
        reps = max(1, int(np.ceil((secs * 1000.0) / float(subframe_ms))))
        for _ in range(reps):
            imgs.append(im_rgb)     # no copy para ahorrar RAM
            durations.append(int(subframe_ms))
    if not imgs:
        return
    first, rest = imgs[0], imgs[1:]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False
    )

# ------ Estilo / basemap-only ------
def style_axes(ax, cb=None, title_text=None):
    if title_text:
        ax.set_title(title_text, fontsize=TITLE_FS, fontweight="bold", bbox=TITLE_BOX)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlabel(ax.get_xlabel() or "", fontsize=LABEL_FS)
    ax.set_ylabel(ax.get_ylabel() or "", fontsize=LABEL_FS)
    if cb is not None:
        cb.ax.tick_params(labelsize=CB_FS)
        if cb.ax.yaxis.label:
            cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=LABEL_FS)
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.86)

def make_basemap_only_frame(extent, title_text, zoom_hint=None):
    fs = _figsize_from_extent(extent, base_height=FIGSIZE[1])
    fig, ax = plt.subplots(figsize=fs, dpi=DPI)
    fig.patch.set_alpha(0.0); ax.set_facecolor((0,0,0,0))
    if extent is not None:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        add_basemap_smart(ax, extent, zoom_hint=zoom_hint)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    style_axes(ax, None, title_text)
    frame = fig_to_rgba(fig)
    plt.close(fig)
    return frame

# ------------- Render temporal (ETA) -------------
def render_eta_timeseries(da, time_dim, out_path, title=None, vmin=None, vmax=None, zoom_hint=None):
    global BASEMAP_FAILED
    BASEMAP_FAILED = False
    frames, secs = [], []
    T = da.sizes[time_dim]
    ydim, xdim = get_xy_dims(da)
    extent = infer_extent(da, ydim, xdim)
    cmap = cmap_with_alpha("turbo")

    # Intro
    if INTRO_BASEMAP:
        base_title = (title + " • mapa base") if title else "Mapa base"
        for _ in range(max(1, INTRO_BASEMAP_FRAMES)):
            frames.append(make_basemap_only_frame(extent, base_title, zoom_hint=zoom_hint))
            secs.append(float(INTRO_BASEMAP_SECS))

    fs = _figsize_from_extent(extent, base_height=FIGSIZE[1])

    for it in range(T):
        arr = da.isel({time_dim: it}).values
        if CLIP_NEGATIVE_TO_NAN and np.issubdtype(arr.dtype, np.number):
            arr = np.where(arr < 0, np.nan, arr)
        if STRIDE > 1 and arr.ndim >= 2:
            arr = arr[::STRIDE, ::STRIDE]

        fig, ax = plt.subplots(figsize=fs, dpi=DPI)
        fig.patch.set_alpha(0.0); ax.set_facecolor((0,0,0,0))
        if extent is not None:
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            add_basemap_smart(ax, extent, zoom_hint=zoom_hint)
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax,
                           interpolation="nearest", extent=extent, cmap=cmap, aspect='equal')
            ax.set_aspect('equal', adjustable='box')
        else:
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax,
                           interpolation="nearest", cmap=cmap, aspect='equal')
            ax.set_aspect('equal', adjustable='box')

        cb = add_colorbar(im, ax, f"{da.name} (unid.)")

        title_txt = f"{title}\n{time_dim}={it+1}/{T}" if title else None
        style_axes(ax, cb, title_txt)

        if STAMP_E_LABEL:
            ax.text(0.02, 0.02, f"{time_dim}:{it+1}", transform=ax.transAxes,
                    fontsize=STAMP_FS, color="white", bbox=STAMP_BOX)

        frames.append(fig_to_rgba(fig)); secs.append(float(ETA_SECS_PER_FRAME))
        plt.close(fig)

    if frames:
        frames.insert(0, frames[0]); secs.insert(0, HOLD_SECS)
        frames.append(frames[-1]); secs.append(HOLD_SECS)

    save_gif_ultraslow(out_path, frames, secs, subframe_ms=SUBFRAME_MS)
    print(f"[OK] {out_path}")

# ------------- Render escaneo e1..e10 (2D) -------------
def render_escan_por_W(zone, e_to_path, var_candidates, out_path, title_base, is_flux=False, zoom_hint=None):
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
            vname = pick_strict_2d_var(ds, var_candidates)
            if vname is None:
                print(f"[WARN] {zone} {ei}: ninguna variable de {var_candidates} presente")
                arrays.append(None); extents.append(None); continue
            used_name = used_name or vname
            da = ds[vname]
            zdim = find_dim(ds, ["zone", "region", "group", "Z", "W", "w"])
            if zdim:
                idx = next((v for v in ds.coords[zdim].values if norm_W_label(v) == norm_W_label(zone)), None)
                if idx is not None: da = da.sel({zdim: idx})
            # aplana dims extra tamaño 1
            while da.ndim > 2:
                for d in list(da.dims):
                    if d not in X_DIMS + Y_DIMS and da.sizes[d] == 1:
                        da = da.isel({d: 0}); break
                else: break
            ydim, xdim = get_xy_dims(da)
            ext = infer_extent(da, ydim, xdim)
            a = da.values

            # Transparencia
            if is_flux:
                if np.issubdtype(a.dtype, np.number):
                    a = np.where(a <= FLUX_ZERO_EPS, np.nan, a)
            else:
                if CLIP_NEGATIVE_TO_NAN and np.issubdtype(a.dtype, np.number):
                    a = np.where(a <= 0, np.nan, a)

            arrays.append(a); extents.append(ext)
            print(f"[INFO] {zone} {ei}: usando variable '{vname}'")
        finally:
            try:
                if ds is not None: ds.close()
            except Exception:
                pass

    vmin, vmax = vrange_from_arrays(arrays, pct=PERC)
    extent = next((e for e in extents if e is not None), None)
    cmap = cmap_with_alpha("turbo")
    fs = _figsize_from_extent(extent, base_height=FIGSIZE[1])

    frames, secs = [], []

    if INTRO_BASEMAP:
        for _ in range(max(1, INTRO_BASEMAP_FRAMES)):
            frames.append(make_basemap_only_frame(extent, f"{title_base} • mapa base", zoom_hint=zoom_hint))
            secs.append(float(INTRO_BASEMAP_SECS))

    for idx, ei in enumerate([f"e{i}" for i in range(1,11)]):
        arr = arrays[idx]
        fig, ax = plt.subplots(figsize=fs, dpi=DPI)
        fig.patch.set_alpha(0.0); ax.set_facecolor((0,0,0,0))

        if arr is None:
            ax.text(0.5,0.5,f"{ei} (sin datos)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=TITLE_FS, bbox=TITLE_BOX, color="black")
            style_axes(ax, None, f"{title_base} • {ei}")
            if STAMP_E_LABEL:
                ax.text(0.02, 0.02, ei, transform=ax.transAxes,
                        fontsize=STAMP_FS, color="white", bbox=STAMP_BOX)
            ax.set_aspect('equal', adjustable='box')
            frames.append(fig_to_rgba(fig)); secs.append(float(ESCAN_SECS_PER_FRAME))
            plt.close(fig); continue

        if STRIDE > 1 and arr.ndim >= 2:
            arr = arr[::STRIDE, ::STRIDE]

        if extent is not None:
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            add_basemap_smart(ax, extent, zoom_hint=zoom_hint)
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest",
                           extent=extent, cmap=cmap, aspect='equal')
            ax.set_aspect('equal', adjustable='box')
        else:
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax,
                           interpolation="nearest", cmap=cmap, aspect='equal')
            ax.set_aspect('equal', adjustable='box')

        cb = add_colorbar(im, ax, (used_name or "var") + " (unid.)")

        style_axes(ax, cb, f"{title_base} • {ei}")
        if STAMP_E_LABEL:
            ax.text(0.02, 0.02, ei, transform=ax.transAxes,
                    fontsize=STAMP_FS, color="white", bbox=STAMP_BOX)

        frames.append(fig_to_rgba(fig)); secs.append(float(ESCAN_SECS_PER_FRAME))
        plt.close(fig)

    if frames:
        frames.insert(0, frames[0]); secs.insert(0, HOLD_SECS)
        frames.append(frames[-1]); secs.append(HOLD_SECS)

    save_gif_ultraslow(out_path, frames, secs, subframe_ms=SUBFRAME_MS)
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

    zone, esc = parse_zone_scen_from_name(src_name)
    zone = zone or "W"; esc = esc or "eXX"

    vmin, vmax = vrange_from_da(da, pct=PERC)
    lvl = parse_level_from_name(src_name)
    level_tag = f"L{lvl}" if lvl is not None else "L?"
    zoom_hint = DEFAULT_ZOOM  # dejamos que el fallback decida
    title = f"{src_name} • {level_tag} • {zone} • {esc} • {var_name}"
    out_name = sanitize(f"{level_tag}_{zone}_{esc}_{var_name}.gif")
    out_path = os.path.join(OUT_DIR, out_name)
    render_eta_timeseries(da, time_dim, out_path, title=title, vmin=vmin, vmax=vmax, zoom_hint=zoom_hint)

# ------------- Agrupar L4 por W y e -------------
def group_l4_by_zone_e(paths):
    groups = {}
    for f in paths:
        name = f.name
        lvl = parse_level_from_name(name)
        if lvl != 4: continue
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

    l4_groups = group_l4_by_zone_e(files)
    for zone, e_to_path in l4_groups.items():
        # zoom sugerido alto; el fallback baja si hace falta
        z_hint = DEFAULT_ZOOM

        out_h = os.path.join(OUT_DIR, sanitize(f"L4_{zone}_maxheight_escan.gif"))
        title_h = f"{zone} • max height (e1..e10)"
        print(f"\n=== Escaneo MAX HEIGHT: {zone} ===")
        render_escan_por_W(zone, e_to_path, MAX_HEIGHT_CANDS, out_h, title_h, is_flux=False, zoom_hint=z_hint)

        out_m = os.path.join(OUT_DIR, sanitize(f"L4_{zone}_momentumflux_escan.gif"))
        title_m = f"{zone} • momentum flux (e1..e10)"
        print(f"=== Escaneo MOMENTUM FLUX: {zone} ===")
        render_escan_por_W(zone, e_to_path, MOM_FLUX_CANDS,  out_m, title_m, is_flux=True,  zoom_hint=z_hint)

    # ETA por archivo/escenario
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
