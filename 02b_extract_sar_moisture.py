"""
Extracts the Sentinel-1 soil-moisture proxy (VH minus VV, dB) over the T-30
and T-15 windows.

COPERNICUS/S1_GRD arrives orbit-corrected, thermal- and border-noise
removed, calibrated to sigma0 and Range-Doppler terrain corrected, which is
why no such step appears below. This script adds: IW mode with VV+VH, a
per-image 30 m focal-mean speckle filter, a residual -30 dB mask, the ratio,
then a temporal median.

The ratio is formed per image BEFORE the temporal reduction -- a median of
ratios is not the ratio of medians.

Both orbit passes are combined. This choice was inherited from an earlier
ablation in the same ROI, but with a different training anchor and point set:
removing its single-orbit lock reduced full-training-set SAR missingness from
217/514 to 12/514 (2.3%). That 2.3% is not a missingness estimate for this
experiment. Here, test-anchor moisture is complete for all 1,362 points. The
cost is mixed local incidence angles, measured in 12_incidence_angle_check.py.

OUTPUT: moisture.csv        REQUIRES GEE AUTH
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


print("[START] 02b - Extracting Sentinel-1 soil-moisture proxy (T-30, T-15)...")
GEE_PROJECT = require_gee_project()

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(ROI_COORDS)


def mask_border_noise(image):
    """Removes low-intensity swath-edge border noise (VV only, matching 03)."""
    return image.updateMask(image.select('VV').gt(-30))


def process_sar_image(image):
    """Per-image speckle filter + VH-VV ratio, computed BEFORE temporal
    reduction (processing order matters -- see 03's docstring)."""
    smoothed = image.focal_mean(radius=30, units='meters')
    ratio = smoothed.select('VH').subtract(smoothed.select('VV')).rename('VH_VV_ratio_dB')
    return image.addBands(ratio)


def moisture_composite(start_date, end_date):
    """Combines BOTH orbit passes so a point covered by only one still
    gets a value (see docstring for why the single-orbit lock was dropped)."""
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

    coll = base.map(mask_border_noise).map(process_sar_image)
    return coll.select('VH_VV_ratio_dB').median(), n_desc + n_asc, f"{n_desc} DESC + {n_asc} ASC"


points_path = POINTS_PATH
if not os.path.exists(points_path):
    print(f"[FATAL] {points_path} not found.")
    print("        Run 01_prepare_points.py first.")
    exit()
pts = pd.read_csv(points_path)
print(f" -> {len(pts)} points loaded")

t30_start, t30_end = window('T-30', ANCHOR)
t15_start, t15_end = window('T-15', ANCHOR)
print(f"    T-30 window: {t30_start.date()}..{t30_end.date()}")
print(f"    T-15 window: {t15_start.date()}..{t15_end.date()}")

img_t30, n_t30, pass_t30 = moisture_composite(t30_start, t30_end)
img_t15, n_t15, pass_t15 = moisture_composite(t15_start, t15_end)
if img_t30 is None or img_t15 is None:
    print(f"[FATAL] No S1 scenes in one or both windows (T-30 n={n_t30}, T-15 n={n_t15}).")
    exit()
print(f"    T-30: {pass_t30} ({n_t30} total), T-15: {pass_t15} ({n_t15} total)")

combo = ee.Image.cat([
    img_t30.rename('Moisture_T30'),
    img_t15.rename('Moisture_T15'),
])

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

COLS = ['Moisture_T30', 'Moisture_T15']
results = []
for feat in sampled['features']:
    p = feat['properties']
    results.append({'row_id': p.get('row_id'), **{c: p.get(c) for c in COLS}})

out_df = pd.DataFrame(results)
n_missing = out_df[COLS].isna().any(axis=1).sum()
print(f"    -> Moisture extracted for {len(out_df)} rows (missing at least one window: {n_missing})")

out_path = os.path.join(OUT_DIR, "moisture.csv")
out_df.to_csv(out_path, index=False)
print(f"[SUCCESS] Written {out_path}")
print("\nNext: run 02c_extract_sar_rvi.py")
