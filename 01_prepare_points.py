"""
Builds the three point sets used by this study.

    Data/train_points.csv     hopper + band sightings   (train target)
    Data/all_points.csv       swarm + adult sightings   (test target)
    Data/presence_points.csv  all locust categories     (optional target)

All three sit on the same ten-week grid from 2020-02-18. Absences use real
FAO 'NO LOCUST' survey records first, topped up with habitat-masked random
background; the Source column records which is which.

Two data hazards, both hit during development:
  - the absence file is DD-MM-YYYY, not ISO. Without dayfirst=True pandas
    drops 59% of rows and misparses the rest.
  - the same field visit is often filed twice. Both sets are deduplicated
    on (X, Y, ObsDate), which removed 27 of 152 training sightings.
"""


import pandas as pd
import numpy as np
import rasterio
import os

from timeline_config import BASE_DIR, DATA_DIR, ROI_COORDS, WEEK_STARTS, week_bounds

print("[START] 01 - Building TEST point set (swarm + adult)...")

MASK_PATH = os.path.join(BASE_DIR, "Wave1_EastAfrica_Copernicus_Habitat_Mask_100m.tif")

# Raw FAO field reports live in source_data/ so this folder runs standalone.
# There is deliberately no fallback to a sibling folder: a silent fallback
# would let the pipeline run on a different copy of the data without saying so.
SRC_DIR = os.path.join(BASE_DIR, "source_data")
ABS_PATH = os.path.join(SRC_DIR, "Desert locusts observation by day (Global).csv")

REQUIRED = ["hoppers_2020.csv", "bands_2020.csv", "adults_2020.csv",
            "swarms_2020.csv", os.path.basename(ABS_PATH)]
missing = [f for f in REQUIRED if not os.path.exists(os.path.join(SRC_DIR, f))]
if not os.path.isdir(SRC_DIR) or missing:
    print(f"[FATAL] source_data/ is incomplete: {SRC_DIR}")
    for f in (missing or REQUIRED):
        print(f"          missing  {f}")
    raise SystemExit(1)

if not os.path.exists(MASK_PATH):
    print(f"[FATAL] Habitat mask not found: {MASK_PATH}")
    raise SystemExit(1)

# =====================================================================
# POSITIVES -- SWARM + ADULT records (the TEST target)
# =====================================================================
# Both files are ISO-dated (unlike the absence file below, which is
# DD-MM-YYYY) -- confirmed when the dayfirst bug was traced.
parts = []
for fname, src_label in (("swarms_2020.csv", "fao_swarm"),
                         ("adults_2020.csv", "fao_adult")):
    fpath = os.path.join(SRC_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[FATAL] {fpath} not found.")
        exit()
    raw = pd.read_csv(fpath, low_memory=False).rename(columns={'Lon': 'X', 'Lat': 'Y'})
    # Narrow to the three columns we need BEFORE adding any, otherwise pandas
    # emits a fragmentation PerformanceWarning -- these FAO files carry ~150
    # columns and assigning into that width is what triggers it.
    d = pd.DataFrame({
        'X': raw['X'],
        'Y': raw['Y'],
        'ObsDate': pd.to_datetime(raw['Observation Date'], errors='coerce'),
        'Source': src_label,
    })
    d = d.dropna(subset=['X', 'Y', 'ObsDate'])
    d = d[(d.X >= ROI_COORDS[0]) & (d.X <= ROI_COORDS[2]) &
          (d.Y >= ROI_COORDS[1]) & (d.Y <= ROI_COORDS[3])].reset_index(drop=True)
    print(f" -> {len(d):>4} {src_label} records inside ROI")
    parts.append(d)

pos = pd.concat(parts, ignore_index=True)
n_before = len(pos)
# Deduplicate ACROSS and WITHIN both files. 56 of the 79 in-ROI adult records
# share an exact coordinate+date with a swarm record (the same field report
# filed under two categories), and the swarm file itself carries internal
# duplicates. Counting either twice would inflate every recall denominator.
# 'first' keeps the swarm label where both exist, since swarms are the
# primary target and the more specific category.
pos = pos.drop_duplicates(subset=['X', 'Y', 'ObsDate'], keep='first').reset_index(drop=True)
print(f" -> {len(pos)} unique swarm+adult positives after dedup "
      f"({n_before - len(pos)} duplicate coordinate+date pairs removed)")

# =====================================================================
# REAL ABSENCES -- FAO 'NO LOCUST' survey records (DAILY, not monthly --
# the "monthly" claim was an artefact of the date-parsing bug, see docstring)
# =====================================================================
real_abs = pd.DataFrame(columns=['X', 'Y', 'ObsDate'])
if os.path.exists(ABS_PATH):
    a = pd.read_csv(ABS_PATH, low_memory=False)
    # DD-MM-YYYY in this file, unlike every other source (which is ISO).
    # Without dayfirst=True pandas silently drops 59% of rows as NaT and
    # MISPARSES the rest -- see the note in this file's docstring.
    a['ObsDate'] = pd.to_datetime(a['Observation Date'], errors='coerce', dayfirst=True)
    a = a.dropna(subset=['lat', 'lon', 'ObsDate'])
    a = a[(a.Category == 'NO LOCUST') &
          (a.lon >= ROI_COORDS[0]) & (a.lon <= ROI_COORDS[2]) &
          (a.lat >= ROI_COORDS[1]) & (a.lat <= ROI_COORDS[3])]
    real_abs = a.rename(columns={'lon': 'X', 'lat': 'Y'})[['X', 'Y', 'ObsDate']]
    print(f" -> {len(real_abs)} real 'NO LOCUST' survey absences inside ROI")
else:
    print(f" -> [WARN] {ABS_PATH} not found; all negatives will be synthetic.")


def masked_background(n, seed):
    coords, rng = [], np.random.RandomState(seed)
    with rasterio.open(MASK_PATH) as src:
        while len(coords) < n:
            lon = rng.uniform(ROI_COORDS[0], ROI_COORDS[2])
            lat = rng.uniform(ROI_COORDS[1], ROI_COORDS[3])
            try:
                if next(src.sample([(lon, lat)]))[0] == 1:
                    coords.append([lon, lat])
            except (IndexError, StopIteration):
                continue
    return pd.DataFrame(coords, columns=['X', 'Y'])


rows = []
print(f"\n {'week':<5}{'dates':<26}{'swarm':>7}{'adult':>7}{'pos':>6}{'real abs':>10}{'synth':>7}")
print(" " + "-" * 63)

for i, ws in enumerate(WEEK_STARTS):
    wk = i + 1
    start, end = week_bounds(wk)

    p = pos[(pos.ObsDate >= start) & (pos.ObsDate <= end)]
    n_pos = len(p)
    n_sw = int((p.Source == 'fao_swarm').sum())
    n_ad = int((p.Source == 'fao_adult').sum())
    if n_pos == 0:
        print(f" W{wk:<4}{str(start.date()) + '..' + str(end.date()):<26}"
              f"{0:>7}{0:>7}{0:>6}{'-':>10}{'-':>7}  [skipped]")
        continue

    p_out = p[['X', 'Y', 'ObsDate', 'Source']].copy()   # Source = fao_swarm | fao_adult
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
        bg = masked_background(n_synth, seed=200 + wk)
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
          f"{n_sw:>7}{n_ad:>7}{n_pos:>6}{n_real:>10}{n_synth:>7}")

all_points = pd.concat(rows, ignore_index=True)
all_points.insert(0, 'row_id', range(len(all_points)))
all_points = all_points[['row_id', 'X', 'Y', 'Presence', 'Week', 'WeekStart', 'ObsDate', 'Source']]

out_path = os.path.join(DATA_DIR, "all_points.csv")
all_points.to_csv(out_path, index=False)

print("\n Totals:")
print(f"   swarm positives      : {int((all_points.Source == 'fao_swarm').sum())}")
print(f"   adult positives      : {int((all_points.Source == 'fao_adult').sum())}")
print(f"   ALL positives        : {int((all_points.Presence == 1).sum())}")
print(f"   real FAO absences    : {int((all_points.Source == 'fao_no_locust').sum())}")
print(f"   synthetic background : {int((all_points.Source == 'synthetic_background').sum())}")
print(f"   TOTAL                : {len(all_points)}")
print(f"\n[SUCCESS] Written {out_path}")
print("\nNext: run 02a_extract_era5.py")


# #####################################################################
# SECOND POINT SET -- the TRAINING target (hopper + band)
# #####################################################################
print("\n[START] 01 - Building TRAIN point set (hopper + band)...")

# =====================================================================
# POSITIVES -- FAO hopper + band records
# =====================================================================
parts = []
for fname, src_label in (("hoppers_2020.csv", "fao_hopper"),
                         ("bands_2020.csv", "fao_band")):
    fpath = os.path.join(SRC_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[FATAL] {fpath} not found.")
        raise SystemExit(1)
    raw = pd.read_csv(fpath, low_memory=False).rename(columns={'Lon': 'X', 'Lat': 'Y'})
    # Narrow before adding columns -- these FAO files carry ~150 columns and
    # assigning into that width triggers a pandas fragmentation warning.
    d = pd.DataFrame({
        'X': raw['X'],
        'Y': raw['Y'],
        'ObsDate': pd.to_datetime(raw['Observation Date'], errors='coerce'),
        'Source': src_label,
    })
    d = d.dropna(subset=['X', 'Y', 'ObsDate'])
    d = d[(d.X >= ROI_COORDS[0]) & (d.X <= ROI_COORDS[2]) &
          (d.Y >= ROI_COORDS[1]) & (d.Y <= ROI_COORDS[3])].reset_index(drop=True)
    print(f" -> {len(d):>4} {src_label} records inside ROI")
    parts.append(d)

pos = pd.concat(parts, ignore_index=True)
n_before = len(pos)
pos = pos.drop_duplicates(subset=['X', 'Y', 'ObsDate'], keep='first').reset_index(drop=True)
print(f" -> {len(pos)} unique hopper+band positives after dedup "
      f"({n_before - len(pos)} duplicate coordinate+date pairs removed)")

# =====================================================================
# REAL ABSENCES -- FAO 'NO LOCUST' survey records (DAILY; see docstring)
# =====================================================================
real_abs = pd.DataFrame(columns=['X', 'Y', 'ObsDate'])
if os.path.exists(ABS_PATH):
    a = pd.read_csv(ABS_PATH, low_memory=False)
    # DD-MM-YYYY here, unlike every other source. dayfirst=True is required.
    a['ObsDate'] = pd.to_datetime(a['Observation Date'], errors='coerce', dayfirst=True)
    a = a.dropna(subset=['lat', 'lon', 'ObsDate'])
    a = a[(a.Category == 'NO LOCUST') &
          (a.lon >= ROI_COORDS[0]) & (a.lon <= ROI_COORDS[2]) &
          (a.lat >= ROI_COORDS[1]) & (a.lat <= ROI_COORDS[3])]
    real_abs = a.rename(columns={'lon': 'X', 'lat': 'Y'})[['X', 'Y', 'ObsDate']]
    print(f" -> {len(real_abs)} real 'NO LOCUST' survey absences inside ROI")
else:
    print(f" -> [WARN] {ABS_PATH} not found; all negatives will be synthetic.")


def masked_background(n, seed):
    """Seeded uniform points inside the Copernicus habitat mask."""
    coords, rng = [], np.random.RandomState(seed)
    with rasterio.open(MASK_PATH) as src:
        while len(coords) < n:
            lon = rng.uniform(ROI_COORDS[0], ROI_COORDS[2])
            lat = rng.uniform(ROI_COORDS[1], ROI_COORDS[3])
            try:
                if next(src.sample([(lon, lat)]))[0] == 1:
                    coords.append([lon, lat])
            except (IndexError, StopIteration):
                continue
    return pd.DataFrame(coords, columns=['X', 'Y'])


# =====================================================================
# BUILD WEEK BY WEEK
# =====================================================================
rows = []
print(f"\n {'week':<5}{'dates':<26}{'hopper':>8}{'band':>6}{'pos':>6}{'real abs':>10}{'synth':>7}")
print(" " + "-" * 63)

for i, ws in enumerate(WEEK_STARTS):
    wk = i + 1
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

    # Synthetic only tops up a shortfall against this week's positive count.
    n_synth = max(0, n_pos - n_real)
    bg = None
    if n_synth:
        bg = masked_background(n_synth, seed=100 + wk)
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

out_path = os.path.join(DATA_DIR, "train_points.csv")
all_points.to_csv(out_path, index=False)

print("\n Totals:")
print(f"   hopper positives     : {int((all_points.Source == 'fao_hopper').sum())}")
print(f"   band positives       : {int((all_points.Source == 'fao_band').sum())}")
print(f"   ALL positives        : {int((all_points.Presence == 1).sum())}")
print(f"   real FAO absences    : {int((all_points.Source == 'fao_no_locust').sum())}")
print(f"   synthetic background : {int((all_points.Source == 'synthetic_background').sum())}")
print(f"   TOTAL                : {len(all_points)}")
print(f"\n[SUCCESS] Written {out_path}")
print("\nNext: run 02a-02d and 03 with --set train")


# #####################################################################
# THIRD POINT SET -- "PRESENCE": every locust category together
# #####################################################################
# This is the target the published literature uses (Klein et al. AUC 0.761;
# Dynamic Forecast AUC 0.767), so it is what makes this study directly
# comparable to them. It is an EASIER target than swarms, for a reason worth
# stating: the features represent environmental suitability associated with
# breeding habitat, and hoppers/bands hatch out of that habitat. Swarms may fly
# in from elsewhere, so an observed swarm location does not establish landing
# or subsequent breeding.
#
# CAVEAT, and it should be reported: hoppers sit still for weeks, so an April
# hopper near a March hopper may be the SAME insects. Part of any gain here is
# re-detection, not forecasting. Most published presence models share this
# limitation without controlling for it.
#
# OUTPUT: Data/presence_points.csv
print("\n[START] 01 - Building PRESENCE point set (hopper + band + adult + swarm)...")

parts = []
for fname, src_label in (("hoppers_2020.csv", "fao_hopper"),
                         ("bands_2020.csv", "fao_band"),
                         ("adults_2020.csv", "fao_adult"),
                         ("swarms_2020.csv", "fao_swarm")):
    raw = pd.read_csv(os.path.join(SRC_DIR, fname), low_memory=False).rename(
        columns={'Lon': 'X', 'Lat': 'Y'})
    d = pd.DataFrame({'X': raw['X'], 'Y': raw['Y'],
                      'ObsDate': pd.to_datetime(raw['Observation Date'], errors='coerce'),
                      'Source': src_label})
    d = d.dropna(subset=['X', 'Y', 'ObsDate'])
    d = d[(d.X >= ROI_COORDS[0]) & (d.X <= ROI_COORDS[2]) &
          (d.Y >= ROI_COORDS[1]) & (d.Y <= ROI_COORDS[3])].reset_index(drop=True)
    print(f" -> {len(d):>4} {src_label} records inside ROI")
    parts.append(d)

pos = pd.concat(parts, ignore_index=True)
n_before = len(pos)
pos = pos.drop_duplicates(subset=['X', 'Y', 'ObsDate'], keep='first').reset_index(drop=True)
print(f" -> {len(pos)} unique presence records after dedup "
      f"({n_before - len(pos)} duplicates removed)")

rows = []
print(f"\n {'week':<5}{'dates':<26}{'presence':>10}{'real abs':>10}{'synth':>7}")
print(" " + "-" * 59)
for i, ws in enumerate(WEEK_STARTS):
    wk = i + 1
    start, end = week_bounds(wk)
    p = pos[(pos.ObsDate >= start) & (pos.ObsDate <= end)]
    n_pos = len(p)
    if n_pos == 0:
        continue
    p_out = p[['X', 'Y', 'ObsDate', 'Source']].copy()
    p_out['Presence'] = 1

    ra = real_abs[(real_abs.ObsDate >= start) & (real_abs.ObsDate <= end)]
    ra_out = None
    if len(ra):
        ra_out = ra[['X', 'Y', 'ObsDate']].copy()
        ra_out['Presence'] = 0
        ra_out['Source'] = 'fao_no_locust'

    n_synth = max(0, n_pos - len(ra))
    bg = None
    if n_synth:
        bg = masked_background(n_synth, seed=300 + wk)
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
          f"{n_pos:>10}{len(ra):>10}{n_synth:>7}")

allp = pd.concat(rows, ignore_index=True)
allp.insert(0, 'row_id', range(len(allp)))
allp = allp[['row_id', 'X', 'Y', 'Presence', 'Week', 'WeekStart', 'ObsDate', 'Source']]
out_path = os.path.join(DATA_DIR, "presence_points.csv")
allp.to_csv(out_path, index=False)
print(f"\n   presence records : {int((allp.Presence == 1).sum())}")
print(f"   TOTAL            : {len(allp)}")
print(f"[SUCCESS] Written {out_path}")
