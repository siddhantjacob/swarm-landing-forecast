"""
Builds Data_somaliland/somaliland_points.csv: hopper+band positives in the
Somaliland cluster box, restricted to the same training window and weekly
grid as train_points.csv (W1-W5, 2020-02-18..2020-03-23), plus absences.

Mirrors 01_prepare_points.py's TRAIN point-set logic exactly (same real-vs-
synthetic absence rule, same masked_background sampler, same weekly loop),
with two substitutions: the ROI is SOMALILAND_ROI_COORDS instead of the
original ROI_COORDS, and the habitat mask is Somaliland_Habitat_Mask_100m.tif
(built by S0, and NOT confirmed identical in provenance to the original
mask -- see S0's docstring).

Zero real FAO 'NO LOCUST' records fall inside this box in this window (see
somaliland_config.py docstring), so every absence produced here is
synthetic. This is printed loudly below and should be reported alongside
the AUC/recall numbers, not left implicit.
"""

import pandas as pd
import numpy as np
import rasterio
import os

from somaliland_config import (SOMALILAND_ROI_COORDS, TRAIN_WEEKS, DATA_DIR,
                               MASK_PATH, SRC_DIR, WEEK_STARTS, week_bounds)

print("[START] S1 - Building Somaliland point set (hopper + band)...")

ABS_PATH = os.path.join(SRC_DIR, "Desert locusts observation by day (Global).csv")

REQUIRED = ["hoppers_2020.csv", "bands_2020.csv", os.path.basename(ABS_PATH)]
missing = [f for f in REQUIRED if not os.path.exists(os.path.join(SRC_DIR, f))]
if missing:
    print(f"[FATAL] source_data/ is missing: {missing}")
    raise SystemExit(1)

if not os.path.exists(MASK_PATH):
    print(f"[FATAL] {MASK_PATH} not found. Run S0_build_habitat_mask.py first.")
    raise SystemExit(1)

# =====================================================================
# POSITIVES -- hopper + band, inside the Somaliland box
# =====================================================================
parts = []
for fname, src_label in (("hoppers_2020.csv", "fao_hopper"),
                         ("bands_2020.csv", "fao_band")):
    raw = pd.read_csv(os.path.join(SRC_DIR, fname), low_memory=False).rename(
        columns={'Lon': 'X', 'Lat': 'Y'})
    d = pd.DataFrame({
        'X': raw['X'], 'Y': raw['Y'],
        'ObsDate': pd.to_datetime(raw['Observation Date'], errors='coerce'),
        'Source': src_label,
    })
    d = d.dropna(subset=['X', 'Y', 'ObsDate'])
    d = d[(d.X >= SOMALILAND_ROI_COORDS[0]) & (d.X <= SOMALILAND_ROI_COORDS[2]) &
          (d.Y >= SOMALILAND_ROI_COORDS[1]) & (d.Y <= SOMALILAND_ROI_COORDS[3])].reset_index(drop=True)
    print(f" -> {len(d):>4} {src_label} records inside Somaliland ROI")
    parts.append(d)

pos = pd.concat(parts, ignore_index=True)
n_before = len(pos)
pos = pos.drop_duplicates(subset=['X', 'Y', 'ObsDate'], keep='first').reset_index(drop=True)
print(f" -> {len(pos)} unique hopper+band positives after dedup "
      f"({n_before - len(pos)} duplicates removed)")

# =====================================================================
# REAL ABSENCES -- FAO 'NO LOCUST', same box/window (expected: 0)
# =====================================================================
real_abs = pd.DataFrame(columns=['X', 'Y', 'ObsDate'])
if os.path.exists(ABS_PATH):
    a = pd.read_csv(ABS_PATH, low_memory=False)
    a['ObsDate'] = pd.to_datetime(a['Observation Date'], errors='coerce', dayfirst=True)
    a = a.dropna(subset=['lat', 'lon', 'ObsDate'])
    a = a[(a.Category == 'NO LOCUST') &
          (a.lon >= SOMALILAND_ROI_COORDS[0]) & (a.lon <= SOMALILAND_ROI_COORDS[2]) &
          (a.lat >= SOMALILAND_ROI_COORDS[1]) & (a.lat <= SOMALILAND_ROI_COORDS[3])]
    real_abs = a.rename(columns={'lon': 'X', 'lat': 'Y'})[['X', 'Y', 'ObsDate']]
    print(f" -> {len(real_abs)} real 'NO LOCUST' survey absences inside Somaliland ROI")
    if len(real_abs) == 0:
        print("    [NOTE] 0 real absences here -- every negative below is synthetic. "
              "Report this alongside any AUC/recall number from this test.")


def masked_background(n, seed):
    coords, rng = [], np.random.RandomState(seed)
    with rasterio.open(MASK_PATH) as src:
        tries = 0
        while len(coords) < n:
            tries += 1
            if tries > 200000:
                print(f"[FATAL] Could not find {n} habitat-masked points after "
                      f"{tries} tries. Check {MASK_PATH} isn't all-zero.")
                raise SystemExit(1)
            lon = rng.uniform(SOMALILAND_ROI_COORDS[0], SOMALILAND_ROI_COORDS[2])
            lat = rng.uniform(SOMALILAND_ROI_COORDS[1], SOMALILAND_ROI_COORDS[3])
            try:
                if next(src.sample([(lon, lat)]))[0] == 1:
                    coords.append([lon, lat])
            except (IndexError, StopIteration):
                continue
    return pd.DataFrame(coords, columns=['X', 'Y'])


rows = []
print(f"\n {'week':<5}{'dates':<26}{'hopper':>8}{'band':>6}{'pos':>6}{'real abs':>10}{'synth':>7}")
print(" " + "-" * 63)

for i, ws in enumerate(WEEK_STARTS):
    wk = i + 1
    if wk not in TRAIN_WEEKS:
        continue
    start, end = week_bounds(wk)

    p = pos[(pos.ObsDate >= start) & (pos.ObsDate <= end)]
    n_pos = len(p)
    n_h = int((p.Source == 'fao_hopper').sum())
    n_b = int((p.Source == 'fao_band').sum())
    if n_pos == 0:
        print(f" W{wk:<4}{str(start.date()) + '..' + str(end.date()):<26}"
              f"{0:>8}{0:>6}{0:>6}{'-':>10}{'-':>7}  [skipped]")
        continue

    p_out = p[['X', 'Y', 'ObsDate', 'Source']].copy()
    p_out['Presence'] = 1

    ra = real_abs[(real_abs.ObsDate >= start) & (real_abs.ObsDate <= end)]
    n_real = len(ra)
    ra_out = None
    if n_real:
        ra_out = ra[['X', 'Y', 'ObsDate']].copy()
        ra_out['Presence'] = 0
        ra_out['Source'] = 'fao_no_locust'

    n_synth = max(0, n_pos - n_real)
    bg = None
    if n_synth:
        bg = masked_background(n_synth, seed=900 + wk)
        bg['ObsDate'] = pd.NaT
        bg['Presence'] = 0
        bg['Source'] = 'synthetic_background'

    for part in (p_out, ra_out, bg):
        if part is not None and len(part):
            part = part.copy()
            part['Week'] = wk
            part['WeekStart'] = start.date()
            rows.append(part)

    print(f" W{wk:<4}{str(start.date()) + '..' + str(end.date()):<26}"
          f"{n_h:>8}{n_b:>6}{n_pos:>6}{n_real:>10}{n_synth:>7}")

all_points = pd.concat(rows, ignore_index=True)
all_points.insert(0, 'row_id', range(len(all_points)))
all_points = all_points[['row_id', 'X', 'Y', 'Presence', 'Week', 'WeekStart', 'ObsDate', 'Source']]

out_path = os.path.join(DATA_DIR, "somaliland_points.csv")
all_points.to_csv(out_path, index=False)

print("\n Totals:")
print(f"   hopper positives     : {int((all_points.Source == 'fao_hopper').sum())}")
print(f"   band positives       : {int((all_points.Source == 'fao_band').sum())}")
print(f"   ALL positives        : {int((all_points.Presence == 1).sum())}")
print(f"   real FAO absences    : {int((all_points.Source == 'fao_no_locust').sum())}")
print(f"   synthetic background : {int((all_points.Source == 'synthetic_background').sum())}")
print(f"   TOTAL                : {len(all_points)}")
print(f"\n[SUCCESS] Written {out_path}")
print("\nNext: run S2_extract_features.py")
