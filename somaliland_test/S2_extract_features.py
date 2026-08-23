"""
Extracts all 9 features (5 ERA5-Land + 4 Sentinel-1) for the Somaliland
points, at the TRAIN anchor (2020-02-03) -- these are hopper+band sightings
within the training window, so they get the same anchor and feature windows
train_points.csv does. Logic is identical to 02a/02b/02c, only the ROI and
point file change.

OUTPUT: Data_somaliland/era5.csv, moisture.csv, rvi.csv, features_all.csv
        REQUIRES GEE AUTH
"""

import ee
import pandas as pd
import os

from somaliland_config import (SOMALILAND_ROI_COORDS, DATA_DIR, ANCHOR_DATE,
                               window, ALL_FEATURES, require_gee_project)

print("[START] S2 - Extracting ERA5 + Sentinel-1 features for Somaliland points...")
print(f"    anchor {ANCHOR_DATE.date()} (TRAIN anchor -- unchanged)")
GEE_PROJECT = require_gee_project()

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(SOMALILAND_ROI_COORDS)

points_path = os.path.join(DATA_DIR, "somaliland_points.csv")
if not os.path.exists(points_path):
    print(f"[FATAL] {points_path} not found. Run S1_prepare_points.py first.")
    raise SystemExit(1)
pts = pd.read_csv(points_path)
print(f" -> {len(pts)} points loaded")

features = [
    ee.Feature(ee.Geometry.Point([float(r['X']), float(r['Y'])]), {'row_id': int(r['row_id'])})
    for _, r in pts.iterrows()
]
fc = ee.FeatureCollection(features)

# ---------------------------------------------------------------------
# ERA5-Land  (mirrors 02a)
# ---------------------------------------------------------------------
print("\n-- ERA5-Land --")
ERA5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')

t45_start, t45_end = window('T-45', ANCHOR_DATE)
t30_start, t30_end = window('T-30', ANCHOR_DATE)
t15_start, t15_end = window('T-15', ANCHOR_DATE)

coll_t45 = ERA5.filterBounds(roi).filterDate(str(t45_start.date()), str(t45_end.date()))
coll_t30 = ERA5.filterBounds(roi).filterDate(str(t30_start.date()), str(t30_end.date()))
coll_t15 = ERA5.filterBounds(roi).filterDate(str(t15_start.date()), str(t15_end.date()))

era5_combo = ee.Image.cat([
    coll_t45.select('total_precipitation_sum').sum().rename('Precip_T45'),
    coll_t30.select('total_precipitation_sum').sum().rename('Precip'),
    coll_t30.select('temperature_2m').mean().rename('Temp'),
    coll_t30.select('soil_temperature_level_1').mean().rename('SoilTemp_T30'),
    coll_t15.select('soil_temperature_level_1').mean().rename('SoilTemp_T15'),
])

era5_sampled = era5_combo.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=100).getInfo()
ERA5_COLS = ['Precip_T45', 'Precip', 'Temp', 'SoilTemp_T30', 'SoilTemp_T15']
era5_rows = [{'row_id': f['properties'].get('row_id'),
             **{c: f['properties'].get(c) for c in ERA5_COLS}}
            for f in era5_sampled['features']]
era5_df = pd.DataFrame(era5_rows)
era5_df.to_csv(os.path.join(DATA_DIR, "era5.csv"), index=False)
print(f"    -> {len(era5_df)} rows, "
      f"{era5_df[ERA5_COLS].isna().any(axis=1).sum()} with a missing value")

# ---------------------------------------------------------------------
# Sentinel-1 moisture proxy (VH-VV)  (mirrors 02b)
# ---------------------------------------------------------------------
print("\n-- Sentinel-1 moisture proxy (VH-VV) --")


def mask_border_noise(image):
    return image.updateMask(image.select('VV').gt(-30))


def process_sar_image(image):
    smoothed = image.focal_mean(radius=30, units='meters')
    ratio = smoothed.select('VH').subtract(smoothed.select('VV')).rename('VH_VV_ratio_dB')
    return image.addBands(ratio)


def moisture_composite(start_date, end_date):
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


img_t30, n_t30, pass_t30 = moisture_composite(t30_start, t30_end)
img_t15, n_t15, pass_t15 = moisture_composite(t15_start, t15_end)
if img_t30 is None or img_t15 is None:
    print(f"[FATAL] No S1 scenes over the Somaliland ROI in one or both windows "
          f"(T-30 n={n_t30}, T-15 n={n_t15}). Cannot continue -- Sentinel-1 "
          f"acquisition coverage may not extend here the way it does over the "
          f"original ROI. Check manually in the GEE code editor before assuming "
          f"this is a bug.")
    raise SystemExit(1)
print(f"    T-30: {pass_t30} ({n_t30} total), T-15: {pass_t15} ({n_t15} total)")

moisture_combo = ee.Image.cat([img_t30.rename('Moisture_T30'), img_t15.rename('Moisture_T15')])
moisture_sampled = moisture_combo.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=100).getInfo()
MOIST_COLS = ['Moisture_T30', 'Moisture_T15']
moist_rows = [{'row_id': f['properties'].get('row_id'),
              **{c: f['properties'].get(c) for c in MOIST_COLS}}
             for f in moisture_sampled['features']]
moist_df = pd.DataFrame(moist_rows)
moist_df.to_csv(os.path.join(DATA_DIR, "moisture.csv"), index=False)
print(f"    -> {len(moist_df)} rows, "
      f"{moist_df[MOIST_COLS].isna().any(axis=1).sum()} with a missing value")

# ---------------------------------------------------------------------
# Sentinel-1 RVI  (mirrors 02c)
# ---------------------------------------------------------------------
print("\n-- Sentinel-1 RVI --")


def smooth_and_mask(image):
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


rvi_bands = []
for win_name, out_name in [('T-45', 'RVI_T45'), ('T-30', 'RVI_T30'), ('T0', 'RVI_T0')]:
    s, e = window(win_name, ANCHOR_DATE)
    img, n_scenes, orbit = rvi_composite(s, e)
    if img is None:
        print(f"[FATAL] No S1 scenes in {win_name} window ({s.date()}..{e.date()}).")
        raise SystemExit(1)
    print(f"    {win_name:5s} {s.date()}..{e.date()}: {orbit} ({n_scenes} total) -> {out_name}")
    rvi_bands.append(img.rename(out_name))

rvi_combo = ee.Image.cat(rvi_bands)
rvi_sampled = rvi_combo.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=100).getInfo()
RVI_COLS = ['RVI_T45', 'RVI_T30', 'RVI_T0']
rvi_rows = [{'row_id': f['properties'].get('row_id'),
            **{c: f['properties'].get(c) for c in RVI_COLS}}
           for f in rvi_sampled['features']]
rvi_df = pd.DataFrame(rvi_rows)
rvi_df.to_csv(os.path.join(DATA_DIR, "rvi.csv"), index=False)
for c in RVI_COLS:
    print(f"    -> {c}: {rvi_df[c].isna().sum()} missing of {len(rvi_df)}")

# ---------------------------------------------------------------------
# MERGE  (mirrors 03_merge_features.py)
# ---------------------------------------------------------------------
print("\n-- Merging --")
df = pts.merge(era5_df, on='row_id', how='left') \
        .merge(moist_df, on='row_id', how='left') \
        .merge(rvi_df, on='row_id', how='left')
df['RVI_Phenology_Delta'] = df['RVI_T30'] - df['RVI_T45']
complete = df[ALL_FEATURES].notna().all(axis=1)
print(f" -> {len(df)} rows; {int(complete.sum())} complete on all {len(ALL_FEATURES)} features")

out_path = os.path.join(DATA_DIR, "features_all.csv")
df.to_csv(out_path, index=False)
print(f"\n[SUCCESS] Written {out_path}")
print("\nNext: run S3_merge_and_test.py")
