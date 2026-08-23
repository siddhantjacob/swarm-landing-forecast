"""
Builds a binary habitat mask for the Somaliland ROI, matching the format of
Wave1_EastAfrica_Copernicus_Habitat_Mask_100m.tif (EPSG:4326, ~100 m, single
band, values 0/1) so S1_prepare_points.py can draw masked synthetic
background the same way 01_prepare_points.py does for the original ROI.

DOCUMENTED ASSUMPTION -- READ BEFORE TRUSTING THE OUTPUT
The script that built the original Wave1 mask is not present in this repo,
so the exact land-cover source and class thresholds it used cannot be
recovered here. This script reconstructs a defensible equivalent instead of
guessing at the original one:

    Source:  COPERNICUS/Landcover/100m/Proba-V-C3/Global (2019 epoch),
             band 'discrete_classification'. Chosen because it is the
             Copernicus product most commonly paired with 100 m Sentinel-1
             work in Earth Engine, matches "Copernicus" + "100m" in the
             original filename, and needs no auth beyond the same GEE
             session 02a-02c already require.
    Habitat = 1 for: bare/sparse vegetation, grassland, shrubland, cropland,
             moss/lichen -- i.e. everything except open water, permanent
             snow/ice, and built-up/urban.
    Habitat = 0 for: water bodies (class 80), permanent snow/ice (70),
             built-up (50).

If this differs from however the original mask was built, the synthetic
absences drawn from it are not on identical footing with the ones the main
pipeline uses. Compare a few known-habitat points from the original ROI
against both masks if this matters for how much weight to put on the
result.

OUTPUT: Somaliland_Habitat_Mask_100m.tif   REQUIRES GEE AUTH
"""

import ee
import os

from somaliland_config import (SOMALILAND_ROI_COORDS, MASK_PATH,
                               require_gee_project)

print("[START] S0 - Building Somaliland habitat mask (Copernicus Global Land Cover 100m)...")
GEE_PROJECT = require_gee_project()

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(SOMALILAND_ROI_COORDS)

lc = (ee.ImageCollection('COPERNICUS/Landcover/100m/Proba-V-C3/Global')
      .filterDate('2019-01-01', '2020-01-01')
      .first()
      .select('discrete_classification'))

NON_HABITAT_CLASSES = [50, 70, 80, 200]  # built-up, snow/ice, water, oceans
mask = lc.remap(NON_HABITAT_CLASSES, [0] * len(NON_HABITAT_CLASSES), 1).rename('habitat').clip(roi)

# Export locally via getDownloadURL (small ROI, no Drive export needed).
import urllib.request

url = mask.getDownloadURL({
    'scale': 100,
    'crs': 'EPSG:4326',
    'region': roi,
    'format': 'GEO_TIFF',
})
print(f"    downloading -> {MASK_PATH}")
urllib.request.urlretrieve(url, MASK_PATH)

import rasterio
import numpy as np
with rasterio.open(MASK_PATH) as src:
    arr = src.read(1)
    print(f"    shape {src.shape}, values {np.unique(arr)}, "
          f"habitat fraction {float((arr == 1).mean()):.1%}")

print(f"[SUCCESS] Written {MASK_PATH}")
print("\nNext: run S1_prepare_points.py")
