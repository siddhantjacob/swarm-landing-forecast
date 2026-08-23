"""
Exports the five ERA5 feature rasters across the full ROI at the test
anchor, for the gridded risk map. Writes to Google Drive; download the
GeoTIFFs into Data/risk_rasters/ before running 06.

No point data is read here, so these exports are unaffected by re-running
the point extraction.

OUTPUT (to Drive): 5 GeoTIFFs       REQUIRES GEE AUTH
"""

import ee
import os
import time

from timeline_config import (DATA_DIR, ROI_COORDS, TEST_ANCHOR_DATE, test_window,
                             require_gee_project)

EXPORT_SCALE_DEG = 0.01          # ~1.1 km; see the resolution note above
# Earth Engine creates this Google Drive folder if it does not exist. Override
# it without editing the script: export GEE_DRIVE_FOLDER=MyFolder
DRIVE_FOLDER = os.environ.get('GEE_DRIVE_FOLDER', 'LocustRiskRasters')
OUT_DIR = os.path.join(DATA_DIR, "risk_rasters")
os.makedirs(OUT_DIR, exist_ok=True)

print("[START] 05 - Exporting ROI-wide ERA5 rasters at the TEST anchor "
      f"({TEST_ANCHOR_DATE.date()})...")
GEE_PROJECT = require_gee_project()

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(ROI_COORDS)
ERA5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')

t45_s, t45_e = test_window('T-45')
t30_s, t30_e = test_window('T-30')
t15_s, t15_e = test_window('T-15')
print(f"    T-45 {t45_s.date()}..{t45_e.date()}   Precip_T45")
print(f"    T-30 {t30_s.date()}..{t30_e.date()}   Precip, Temp, SoilTemp_T30")
print(f"    T-15 {t15_s.date()}..{t15_e.date()}   SoilTemp_T15")

c45 = ERA5.filterBounds(roi).filterDate(str(t45_s.date()), str(t45_e.date()))
c30 = ERA5.filterBounds(roi).filterDate(str(t30_s.date()), str(t30_e.date()))
c15 = ERA5.filterBounds(roi).filterDate(str(t15_s.date()), str(t15_e.date()))

# Same reducers as 02a_extract_era5.py -- sum for precipitation, mean for
# temperatures. Any divergence here would silently invalidate the map.
BANDS = {
    'Precip_T45':   c45.select('total_precipitation_sum').sum(),
    'Precip':       c30.select('total_precipitation_sum').sum(),
    'Temp':         c30.select('temperature_2m').mean(),
    # level_1 = 0-7 cm. MUST match 02a_extract_era5.py -- the model is
    # fitted on that band, so changing depth here alone would silently
    # score the model against a different variable. See 02a's docstring.
    'SoilTemp_T30': c30.select('soil_temperature_level_1').mean(),
    'SoilTemp_T15': c15.select('soil_temperature_level_1').mean(),
}

print(f"\n[INFO] Exporting {len(BANDS)} bands at {EXPORT_SCALE_DEG} deg "
      f"(~{EXPORT_SCALE_DEG * 111:.1f} km) to Drive folder '{DRIVE_FOLDER}'...")

tasks = []
for name, img in BANDS.items():
    task = ee.batch.Export.image.toDrive(
        image=img.rename(name).clip(roi).toFloat(),
        description=f"locust_risk_{name}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=roi,
        scale=int(EXPORT_SCALE_DEG * 111319),   # metres
        crs='EPSG:4326',
        maxPixels=1e10,
        fileFormat='GeoTIFF',
    )
    task.start()
    tasks.append((name, task))
    print(f"    queued  {name:<14} task id {task.id}")

print("\n[INFO] Polling task status (Ctrl-C is safe -- exports continue "
      "server-side)...")
done = set()
while len(done) < len(tasks):
    time.sleep(20)
    for name, t in tasks:
        if name in done:
            continue
        st = t.status().get('state')
        if st in ('COMPLETED', 'FAILED', 'CANCELLED'):
            done.add(name)
            note = "" if st == 'COMPLETED' else f"  <-- {t.status().get('error_message', '')}"
            print(f"    {name:<14} {st}{note}")

print("\n" + "=" * 78)
print(" NEXT STEP -- MANUAL")
print("=" * 78)
print(f"  Download all five .tif files from Google Drive / {DRIVE_FOLDER}/")
print(f"  into: {OUT_DIR}")
print("  Keep the filenames exactly as exported (Precip_T45.tif etc) --")
print("  06_generate_risk_map.py matches on them and will refuse to guess.")
print("\n  Then run: python 06_generate_risk_map.py")
