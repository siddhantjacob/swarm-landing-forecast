"""
Extracts the Radar Vegetation Index, RVI = 4*VH / (VV + VH), over the T-45,
T-30 and T0 windows.

RVI is computed in linear power, not decibels -- a ratio of logarithms has
no physical meaning. Speckle smoothing runs BEFORE the -30 dB mask so the
threshold tests denoised pixels rather than raw ones; masking first let
speckle spikes punch holes in the mask.

Same collection filters, speckle filter, dual-orbit handling and temporal
median as 02b.

OUTPUT: rvi.csv         REQUIRES GEE AUTH
"""

import ee
import pandas as pd
import os

from timeline_config import (DATA_DIR, ROI_COORDS, window, resolve_extraction,
                             require_gee_project)

# ---------------------------------------------------------------------
# ANCHOR SELECTION. Default = TRAIN anchor (2020-02-03),
# writing into Data/. Run with `--anchor test` to extract the SAME points
# at the TEST anchor (2020-03-15) into Data/test_anchor/ instead, leaving
# the training feature files untouched. See timeline_config.py.
# ---------------------------------------------------------------------
POINTS_PATH, ANCHOR, OUT_DIR, ANCHOR_TAG = resolve_extraction()
print(f"    [{ANCHOR_TAG} | anchor {ANCHOR.date()}]")
print(f"    points -> {os.path.basename(POINTS_PATH)}   output -> {OUT_DIR}")


print("[START] 02c - Extracting Sentinel-1 RVI (T-45, T-30, T0)...")
GEE_PROJECT = require_gee_project()

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(ROI_COORDS)


def smooth_and_mask(image):
    """Speckle-smooths (30m focal mean) BEFORE the border-noise mask, so the
    VV/VH > -30dB test runs on noise-reduced pixels, not raw ones."""
    smoothed = image.focal_mean(radius=30, units='meters')
    vv_mask = smoothed.select('VV').gt(-30)
    vh_mask = smoothed.select('VH').gt(-30)
    return smoothed.updateMask(vv_mask.And(vh_mask))


def compute_rvi(image):
    vv_lin = ee.Image(10).pow(image.select('VV').divide(10))
    vh_lin = ee.Image(10).pow(image.select('VH').divide(10))
    rvi = vh_lin.multiply(4).divide(vv_lin.add(vh_lin)).rename('RVI')
    return image.addBands(rvi)


def rvi_composite(start_date, end_date):
    """Combines BOTH orbit passes (no single-orbit lock)."""
    base = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(roi)
            .filterDate(str(start_date.date()), str(end_date.date()))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    n_desc = base.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).size().getInfo()
    n_asc = base.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).size().getInfo()
    if n_desc + n_asc == 0:
        return None, 0, None

    coll = base.map(smooth_and_mask).map(compute_rvi)
    return coll.select('RVI').median(), n_desc + n_asc, f"{n_desc} DESC + {n_asc} ASC"


points_path = POINTS_PATH
if not os.path.exists(points_path):
    print(f"[FATAL] {points_path} not found.")
    print("        Run 01_prepare_points.py first.")
    exit()
pts = pd.read_csv(points_path)
print(f" -> {len(pts)} points loaded")

bands = []
for win_name, out_name in [('T-45', 'RVI_T45'), ('T-30', 'RVI_T30'), ('T0', 'RVI_T0')]:
    s, e = window(win_name, ANCHOR)
    img, n_scenes, orbit = rvi_composite(s, e)
    if img is None:
        print(f"[FATAL] No S1 scenes in {win_name} window ({s.date()}..{e.date()}).")
        exit()
    print(f"    {win_name:5s} {s.date()}..{e.date()}: {orbit} ({n_scenes} total) -> {out_name}")
    bands.append(img.rename(out_name))

combo = ee.Image.cat(bands)

features = [
    ee.Feature(ee.Geometry.Point([float(r['X']), float(r['Y'])]), {'row_id': int(r['row_id'])})
    for _, r in pts.iterrows()
]
fc = ee.FeatureCollection(features)

sampled = combo.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.first(),  # multi-band -> band names used automatically, no setOutputs()
    scale=100
).getInfo()

COLS = ['RVI_T45', 'RVI_T30', 'RVI_T0']
results = []
for feat in sampled['features']:
    p = feat['properties']
    results.append({'row_id': p.get('row_id'), **{c: p.get(c) for c in COLS}})

out_df = pd.DataFrame(results)
for c in COLS:
    print(f"    -> {c}: {out_df[c].isna().sum()} missing of {len(out_df)}")

out_path = os.path.join(OUT_DIR, "rvi.csv")
out_df.to_csv(out_path, index=False)
print(f"[SUCCESS] Written {out_path}")
print("\nNext: run 02d_extract_smap.py (optional), then 03_merge_features.py")
