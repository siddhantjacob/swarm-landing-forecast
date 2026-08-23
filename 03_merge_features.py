"""
Joins the extracted feature files to their point set, derives
RVI_Phenology_Delta = RVI_T30 - RVI_T45, and verifies that the half-open
feature interval does not overlap the first label week.

Runs in whichever mode resolve_extraction() selects. smap.csv is optional.

OUTPUT: features_all.csv
"""

import pandas as pd
import os

from timeline_config import (DATA_DIR, ALL_FEATURES, resolve_extraction,
                             window, week_bounds)

POINTS_PATH, ANCHOR, OUT_DIR, TAG = resolve_extraction()
print(f"[START] 03 - Merging features   [{TAG} | anchor {ANCHOR.date()}]")

PIECES = [("era5", ["Precip_T45", "Precip", "Temp", "SoilTemp_T30", "SoilTemp_T15"]),
          ("moisture", ["Moisture_T30", "Moisture_T15"]),
          ("rvi", ["RVI_T45", "RVI_T30", "RVI_T0"])]

# SMAP is optional -- merged only if 02d has been run for this mode.
OPTIONAL = [("smap", ["SMAP_Surface_T15", "SMAP_Root_T15"])]

for name in ('T-45', 'T-30', 'T-15', 'T0'):
    s, e = window(name, ANCHOR)
    print(f"      {name:5s} [{s.date()}, {e.date()})")

if not os.path.exists(POINTS_PATH):
    print(f"[FATAL] {POINTS_PATH} not found. Run 01_prepare_points.py first.")
    raise SystemExit(1)

df = pd.read_csv(POINTS_PATH)
print(f"\n -> base: {len(df)} points")

for suffix, cols in PIECES:
    path = os.path.join(OUT_DIR, f"{suffix}.csv")
    if not os.path.exists(path):
        print(f"[FATAL] {path} not found. Run the 02x extractors in this same mode first.")
        raise SystemExit(1)
    piece = pd.read_csv(path)
    print(f"    {suffix:9s}: {len(piece)} rows, "
          f"{piece[cols].isna().any(axis=1).sum()} with missing value(s)")
    df = df.merge(piece, on='row_id', how='left')

for suffix, cols in OPTIONAL:
    path = os.path.join(OUT_DIR, f"{suffix}.csv")
    if os.path.exists(path):
        piece = pd.read_csv(path)
        print(f"    {suffix:9s}: {len(piece)} rows, "
              f"{piece[cols].isna().any(axis=1).sum()} with missing value(s)   [optional]")
        df = df.merge(piece, on='row_id', how='left')
    else:
        print(f"    {suffix:9s}: not present -- skipped "
              f"(run 02d_extract_smap.py to add it)")

df['RVI_Phenology_Delta'] = df['RVI_T30'] - df['RVI_T45']
complete = df[ALL_FEATURES].notna().all(axis=1)
print(f"    derived  : RVI_Phenology_Delta ({df['RVI_Phenology_Delta'].isna().sum()} missing)")
print(f"\n -> {len(df)} rows; {int(complete.sum())} complete on all {len(ALL_FEATURES)} features")

# --- leakage check: GEE filterDate uses [start, end), with end exclusive ---
t0_end = window('T0', ANCHOR)[1]
first_label_wk = 7 if ('test_anchor' in OUT_DIR or 'presence_anchor' in OUT_DIR) else 1
label_start = week_bounds(first_label_wk)[0]
gap = (label_start - t0_end).days
print(f"\n [CHECK] feature interval ends (exclusive) {t0_end.date()}, "
      f"first label week opens {label_start.date()}  ->  {gap} day separation")
if gap < 0:
    print("[FATAL] Feature window overlaps the labels it is meant to predict.")
    raise SystemExit(1)
if gap == 0:
    print("         adjacent half-open intervals; no temporal overlap")

print("\n Per-week sightings (complete rows only):")
for wk in sorted(df.Week.dropna().unique()):
    sub = df[(df.Week == wk) & complete]
    if len(sub):
        print(f"    W{int(wk):<3}{int((sub.Presence == 1).sum()):>5}")

out_path = os.path.join(OUT_DIR, "features_all.csv")
df.to_csv(out_path, index=False)
print(f"\n[SUCCESS] Written {out_path}")
