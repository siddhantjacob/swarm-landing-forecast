"""
Extracts SMAP L4 surface and root-zone soil moisture over the T-15 window,
giving an L-band retrieval to set against the Sentinel-1 backscatter proxy.

NASA/SMAP/SPL4SMGP/007 is model-assimilated and gap-free, so no masking or
quality filtering is applied and the window mean is taken at the 9 km native
grid. The asset is deprecated in favour of /008.

OUTPUT: smap.csv        REQUIRES GEE AUTH
"""

import ee
import pandas as pd
import os

from timeline_config import (ROI_COORDS, window, resolve_extraction,
                             require_gee_project)

POINTS_PATH, ANCHOR, OUT_DIR, TAG = resolve_extraction()
print(f"    [{TAG} | anchor {ANCHOR.date()}]")
print(f"    points -> {os.path.basename(POINTS_PATH)}   output -> {OUT_DIR}")
print("[START] 02d - Extracting SMAP L4 L-band soil moisture (9 km retrieval)...")
GEE_PROJECT = require_gee_project()

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(ROI_COORDS)
SMAP = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')

if not os.path.exists(POINTS_PATH):
    print(f"[FATAL] {POINTS_PATH} not found.")
    print("        Run 01_prepare_points.py first.")
    raise SystemExit(1)
pts = pd.read_csv(POINTS_PATH)
print(f" -> {len(pts)} points loaded")

# Same T-15 window as the Sentinel-1 moisture feature, so the comparison
# between the two is like-for-like rather than confounded by timing.
t15_start, t15_end = window('T-15', ANCHOR)
print(f"    T-15 window: {t15_start.date()}..{t15_end.date()}  "
      f"(SMAP_Surface_T15, SMAP_Root_T15)")

coll = SMAP.filterBounds(roi).filterDate(str(t15_start.date()), str(t15_end.date()))
n_img = coll.size().getInfo()
print(f"    {n_img} SMAP L4 granules in window (3-hourly, so ~8/day expected)")
if n_img == 0:
    print("[FATAL] No SMAP imagery in this window. Check the date range.")
    raise SystemExit(1)

combo = ee.Image.cat([
    coll.select('sm_surface_wetness').mean().rename('SMAP_Surface_T15'),
    coll.select('sm_rootzone_wetness').mean().rename('SMAP_Root_T15'),
])

features = [
    ee.Feature(ee.Geometry.Point([float(r['X']), float(r['Y'])]), {'row_id': int(r['row_id'])})
    for _, r in pts.iterrows()
]
fc = ee.FeatureCollection(features)

sampled = combo.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.first(),   # multi-band -> band names used automatically
    scale=9000,                   # SMAP L4 native grid
).getInfo()

COLS = ['SMAP_Surface_T15', 'SMAP_Root_T15']
results = []
for feat in sampled['features']:
    p = feat['properties']
    results.append({'row_id': p.get('row_id'), **{c: p.get(c) for c in COLS}})

out_df = pd.DataFrame(results)
n_missing = out_df[COLS].isna().any(axis=1).sum()
print(f"    -> SMAP extracted for {len(out_df)} rows "
      f"(missing at least one value: {n_missing})")

out_path = os.path.join(OUT_DIR, "smap.csv")
out_df.to_csv(out_path, index=False)
print(f"[SUCCESS] Written {out_path}")
print("\nNext: re-run 03_merge_features.py in the same mode, then")
print("      04_train_and_test.py --with-smap")
