"""
Extracts five ERA5-Land weather features at whichever anchor
resolve_extraction() selects (see timeline_config.py).

    Precip_T45    total precipitation, T-45 window
    Precip        total precipitation, T-30
    Temp          mean 2 m air temperature, T-30
    SoilTemp_T30  mean soil temperature, T-30    ERA5-Land level 1,
    SoilTemp_T15  mean soil temperature, T-15    i.e. 0-7 cm

Precip_T45 follows the published 41-64 day lag between rainfall and band
appearance rather than being chosen empirically.

Soil temperature uses level 1 (0-7 cm). Locust eggs sit nearer 5-10 cm,
straddling the level-1/level-2 boundary; level 1 responds fastest to surface
forcing. Any change of depth must be mirrored in 05 and 11, which score the
same fitted model.

reduceRegions samples a five-band ee.Image.cat, so band names are applied
automatically. Do not add setOutputs() -- it errors on multi-band images.

OUTPUT: era5.csv        REQUIRES GEE AUTH
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


print("[START] 02a - Extracting ERA5 weather features (single anchor, all phases)...")
GEE_PROJECT = require_gee_project()

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(ROI_COORDS)
ERA5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')

points_path = POINTS_PATH
if not os.path.exists(points_path):
    print(f"[FATAL] {points_path} not found.")
    print("        Run 01_prepare_points.py first.")
    exit()
pts = pd.read_csv(points_path)
print(f" -> {len(pts)} points loaded")

t45_start, t45_end = window('T-45', ANCHOR)
t30_start, t30_end = window('T-30', ANCHOR)
t15_start, t15_end = window('T-15', ANCHOR)
print(f"    T-45 window: {t45_start.date()}..{t45_end.date()}  (Precip_T45)")
print(f"    T-30 window: {t30_start.date()}..{t30_end.date()}  (Precip, Temp, SoilTemp_T30)")
print(f"    T-15 window: {t15_start.date()}..{t15_end.date()}  (SoilTemp_T15)")

coll_t45 = ERA5.filterBounds(roi).filterDate(str(t45_start.date()), str(t45_end.date()))
coll_t30 = ERA5.filterBounds(roi).filterDate(str(t30_start.date()), str(t30_end.date()))
coll_t15 = ERA5.filterBounds(roi).filterDate(str(t15_start.date()), str(t15_end.date()))

combo = ee.Image.cat([
    coll_t45.select('total_precipitation_sum').sum().rename('Precip_T45'),
    coll_t30.select('total_precipitation_sum').sum().rename('Precip'),
    coll_t30.select('temperature_2m').mean().rename('Temp'),
    coll_t30.select('soil_temperature_level_1').mean().rename('SoilTemp_T30'),
    coll_t15.select('soil_temperature_level_1').mean().rename('SoilTemp_T15'),
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

COLS = ['Precip_T45', 'Precip', 'Temp', 'SoilTemp_T30', 'SoilTemp_T15']
results = []
for feat in sampled['features']:
    p = feat['properties']
    results.append({'row_id': p.get('row_id'), **{c: p.get(c) for c in COLS}})

out_df = pd.DataFrame(results)
n_missing = out_df[COLS].isna().any(axis=1).sum()
print(f"    -> ERA5 extracted for {len(out_df)} rows (missing at least one value: {n_missing})")

out_path = os.path.join(OUT_DIR, "era5.csv")
out_df.to_csv(out_path, index=False)
print(f"[SUCCESS] Written {out_path}")
print("\nNext: run 02b_extract_sar_moisture.py")
