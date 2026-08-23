"""
Re-runs the model-against-baseline comparison using target-group background
(Phillips et al. 2009, Ecological Applications 19(1):181-197): background
drawn from the pooled footprint of locations FAO demonstrably surveyed, so
presence and background share the same survey effort and the model cannot
score by learning where surveyors went.

A robustness check, not a replacement for the primary analysis. Target-group
negatives sit in genuinely similar habitat, so absolute AUC falls by design;
the quantity of interest is the model-baseline difference.

Background is drawn from the POOLED survey footprint rather than week
matched. W7-W10 contain no NO LOCUST records, so a week-matched pool would
be entirely hopper, band and adult sightings -- the same locations the
distance baseline measures from -- which inverts the baseline and looks like
a result.

A cell holding a swarm in a different week stays eligible as background for
this week, since the question is whether a swarm is present in week w.

OUTPUT: tgb_absence_check.csv, Data/tgb_points.csv    REQUIRES GEE AUTH
"""

import ee
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from timeline_config import (RESULTS_DIR, BASE_DIR, DATA_DIR, ROI_COORDS,
                             TEST_ANCHOR_DATE, ERA5_FEATURES, ALL_FEATURES,
                             week_bounds, window, haversine_km,
                             require_gee_project)

TRAIN_WEEKS = [1, 2, 3, 4, 5]
TEST_WEEKS = [7, 8, 9, 10]
GRID_DEG = 0.1                 # ERA5-Land native cell, for the survey diagnostic
N_BOOT = 2000
RNG = np.random.RandomState(42)

SRC_DIR = os.path.join(BASE_DIR, "source_data")
ABS_FILE = "Desert locusts observation by day (Global).csv"
TRAIN_PATH = os.path.join(DATA_DIR, "train_pool", "features_all.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_anchor", "features_all.csv")

print("[START] 11 - Target-group background absence check\n")
GEE_PROJECT = require_gee_project()
for p, hint in ((TRAIN_PATH, "Run 02a/02b/02c --set train, then 03 --set train."),
                (TEST_PATH, "Run 02a/02b/02c --anchor test, then 03 --anchor test."),
                (SRC_DIR, "source_data/ is missing from this folder.")):
    if not os.path.exists(p):
        print(f"[FATAL] {p} not found.\n        {hint}")
        raise SystemExit(1)


def in_roi(df, xc='X', yc='Y'):
    return df[(df[xc] >= ROI_COORDS[0]) & (df[xc] <= ROI_COORDS[2]) &
              (df[yc] >= ROI_COORDS[1]) & (df[yc] <= ROI_COORDS[3])]


# =====================================================================
# 1. LOAD EVERY FAO RECORD -- this is the survey-effort footprint
# =====================================================================
surveyed = []
for fname, lab in (("hoppers_2020.csv", "hopper"), ("bands_2020.csv", "band"),
                   ("adults_2020.csv", "adult"), ("swarms_2020.csv", "swarm")):
    fp = os.path.join(SRC_DIR, fname)
    if not os.path.exists(fp):
        print(f"  [WARN] {fname} not found -- skipped")
        continue
    raw = pd.read_csv(fp, low_memory=False).rename(columns={'Lon': 'X', 'Lat': 'Y'})
    d = pd.DataFrame({'X': raw['X'], 'Y': raw['Y'],
                      'ObsDate': pd.to_datetime(raw['Observation Date'], errors='coerce'),
                      'Cat': lab})
    surveyed.append(in_roi(d.dropna(subset=['X', 'Y', 'ObsDate'])))

# The NO LOCUST file is DD-MM-YYYY. Without dayfirst=True pandas drops 59% of
# rows and misreads the rest -- this bug has already cost one full run.
fp = os.path.join(SRC_DIR, ABS_FILE)
if os.path.exists(fp):
    a = pd.read_csv(fp, low_memory=False)
    a['ObsDate'] = pd.to_datetime(a['Observation Date'], errors='coerce', dayfirst=True)
    a = a.dropna(subset=['lat', 'lon', 'ObsDate'])
    a = a[a.Category == 'NO LOCUST'].rename(columns={'lon': 'X', 'lat': 'Y'})
    d = a[['X', 'Y', 'ObsDate']].copy(); d['Cat'] = 'no_locust'
    surveyed.append(in_roi(d))

surv = pd.concat(surveyed, ignore_index=True).drop_duplicates(
    subset=['X', 'Y', 'ObsDate', 'Cat']).reset_index(drop=True)
print(f" -> {len(surv)} FAO survey records in ROI across all categories")
for c, n in surv.Cat.value_counts().items():
    print(f"      {c:<12}{n:>6}")

# =====================================================================
# 2. THE DIAGNOSTIC -- how much of the ROI was ever actually surveyed?
# =====================================================================
print("\n" + "=" * 84)
print(" HOW MANY SYNTHETIC ABSENCES ASSERT 'NO LOCUSTS' WHERE NOBODY LOOKED?")
print("=" * 84)
cell = lambda x, y: (np.floor(x / GRID_DEG).astype(int), np.floor(y / GRID_DEG).astype(int))
nx = int(np.ceil((ROI_COORDS[2] - ROI_COORDS[0]) / GRID_DEG))
ny = int(np.ceil((ROI_COORDS[3] - ROI_COORDS[1]) / GRID_DEG))
total_cells = nx * ny
sx, sy = cell(surv.X.values, surv.Y.values)
surveyed_cells = set(zip(sx, sy))
swarm = surv[surv.Cat == 'swarm']
wx, wy = cell(swarm.X.values, swarm.Y.values)
swarm_cells = set(zip(wx, wy))

print(f"  ROI grid cells ({GRID_DEG} deg, ~11 km)      {total_cells:>6}")
print(f"  cells with ANY FAO observation          {len(surveyed_cells):>6}  "
      f"({len(surveyed_cells) / total_cells:.1%})")
print(f"  cells with a swarm record               {len(swarm_cells):>6}")
print(f"  surveyed cells WITHOUT a swarm          {len(surveyed_cells - swarm_cells):>6}"
      f"   <- the TGB pool")
print(f"\n  A uniformly random point therefore lands in a surveyed cell about")
print(f"  {len(surveyed_cells) / total_cells:.0%} of the time. The other "
      f"{1 - len(surveyed_cells) / total_cells:.0%} of synthetic absences assert")
print("  'no swarm here' for locations nobody ever visited. That is absence of")
print("  evidence, not evidence of absence (Phillips et al. 2009).")

# =====================================================================
# 3. BUILD THE TGB BACKGROUND
# =====================================================================
# Background is drawn from the pooled survey footprint, not week-matched.
# W7-W10 hold no NO LOCUST records, so a week-matched pool would be entirely
# hopper/band/adult sightings -- the locations the distance baseline measures
# from -- which inverts the baseline (AUC 0.354) and looks like a result.
# The pooled footprint is 57% NO LOCUST records and represents survey effort.
test_all = pd.read_csv(TEST_PATH)
test_all = test_all[(test_all.Presence == 0) | (test_all.Source == 'fao_swarm')].copy()

# Pooled survey footprint, deduplicated by location at the ERA5 grid scale so
# that repeatedly-visited sites do not dominate.
pool_all = surv.copy()
pool_all['cx'], pool_all['cy'] = cell(pool_all.X.values, pool_all.Y.values)
pool_all = pool_all.drop_duplicates(subset=['cx', 'cy', 'Cat'])
print(f"\n -> pooled survey footprint: {len(pool_all)} distinct location/category "
      f"records across {len(surveyed_cells)} cells")

rows = []
print("\n" + "=" * 84)
print(" TGB BACKGROUND DRAWN FROM THE POOLED SURVEY FOOTPRINT")
print("=" * 84)
print(f" {'week':<6}{'swarms':>8}{'TGB pool':>11}{'used':>7}   composition")
print(" " + "-" * 72)
for wk in TEST_WEEKS:
    s, e = week_bounds(wk)
    n_sw = int((test_all[(test_all.Week == wk)].Presence == 1).sum())
    # Exclude any cell holding a swarm THIS week -- those are positives.
    wk_sw = surv[(surv.Cat == 'swarm') & (surv.ObsDate >= s) & (surv.ObsDate <= e)]
    bad = set(zip(*cell(wk_sw.X.values, wk_sw.Y.values)))
    pool = pool_all[~pd.Series(list(zip(pool_all.cx, pool_all.cy)),
                               index=pool_all.index).isin(bad)]
    pool = pool.drop_duplicates(subset=['cx', 'cy'])
    if len(pool) == 0:
        print(f" W{wk:<5}{n_sw:>8}{0:>11}{0:>7}   [pool empty]")
        continue
    take = pool if len(pool) <= n_sw else pool.sample(n_sw, random_state=42 + wk)
    comp = ", ".join(f"{k} {v}" for k, v in take.Cat.value_counts().items())
    print(f" W{wk:<5}{n_sw:>8}{len(pool):>11}{len(take):>7}   {comp}")
    t = take[['X', 'Y', 'ObsDate', 'Cat']].copy()
    t['Week'] = wk
    t['Presence'] = 0
    t['Source'] = 'tgb_' + t.Cat
    rows.append(t.drop(columns='Cat'))

if not rows:
    print("\n[FATAL] No TGB background could be built for any test week.")
    print("        FAO recorded nothing but swarms in W7-W10 inside this ROI.")
    raise SystemExit(1)

tgb = pd.concat(rows, ignore_index=True)
tgb.insert(0, 'row_id', range(len(tgb)))
tgb.to_csv(os.path.join(DATA_DIR, "tgb_points.csv"), index=False)
print(f"\n -> {len(tgb)} TGB background points written to Data/tgb_points.csv")

# =====================================================================
# 4. EXTRACT ERA5 AT THE TEST ANCHOR FOR THE TGB POINTS
# =====================================================================
# Only the five ERA5 features are needed: the headline configuration is
# ERA5-only, and that is the claim being stress-tested. Self-contained on
# purpose -- this does not touch the main extraction chain or its outputs.
print("\n -> extracting ERA5 at the test anchor for the TGB points...")
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(ROI_COORDS)
ERA5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
t45 = window('T-45', TEST_ANCHOR_DATE)
t30 = window('T-30', TEST_ANCHOR_DATE)
t15 = window('T-15', TEST_ANCHOR_DATE)
c = lambda w: ERA5.filterBounds(roi).filterDate(str(w[0].date()), str(w[1].date()))
combo = ee.Image.cat([
    c(t45).select('total_precipitation_sum').sum().rename('Precip_T45'),
    c(t30).select('total_precipitation_sum').sum().rename('Precip'),
    c(t30).select('temperature_2m').mean().rename('Temp'),
    # level_1 = 0-7 cm. MUST match 02a_extract_era5.py -- the model is
    # fitted on that band, so changing depth here alone would silently
    # score the model against a different variable. See 02a's docstring.
    c(t30).select('soil_temperature_level_1').mean().rename('SoilTemp_T30'),
    c(t15).select('soil_temperature_level_1').mean().rename('SoilTemp_T15'),
])
fc = ee.FeatureCollection(
    [ee.Feature(ee.Geometry.Point([float(r.X), float(r.Y)]), {'row_id': int(r.row_id)})
     for _, r in tgb.iterrows()])
got = combo.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                          scale=100).getInfo()
feat = pd.DataFrame([{'row_id': f['properties'].get('row_id'),
                      **{k: f['properties'].get(k) for k in ERA5_FEATURES}}
                     for f in got['features']])
tgb = tgb.merge(feat, on='row_id', how='left').dropna(subset=ERA5_FEATURES)
print(f"    -> {len(tgb)} TGB points with complete ERA5 features")

# =====================================================================
# 5. SCORE BOTH ABSENCE DEFINITIONS
# =====================================================================
train_all = pd.read_csv(TRAIN_PATH)


def zscore(df, ref, cols):
    out = df.copy()
    for col in cols:
        mu, sd = ref[col].mean(), ref[col].std()
        out[col] = (df[col] - mu) / (sd if sd else 1.0)
    return out


def paired_bootstrap(y, a, b, n=N_BOOT):
    idx = np.arange(len(y)); diffs = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        diffs.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


trz = zscore(train_all, train_all, ALL_FEATURES)
tr = trz[trz.Week.isin(TRAIN_WEEKS)].dropna(subset=ERA5_FEATURES)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                               max_depth=5, random_state=42)
model.fit(tr[ERA5_FEATURES], tr.Presence)
known = train_all[train_all.Week.isin(TRAIN_WEEKS) & (train_all.Presence == 1)]

pos_all = test_all[test_all.Presence == 1]
# Each absence set is standardised against its own pooled distribution, the
# same within-anchor rule used everywhere else in this study.
variants = {
    'synthetic background (primary)': pd.concat(
        [pos_all, test_all[test_all.Presence == 0]], ignore_index=True),
    'target-group background (TGB)': pd.concat(
        [pos_all, tgb], ignore_index=True),
}

print("\n" + "=" * 84)
print(" DOES THE BASELINE'S LEAD SURVIVE THE CHANGE OF ABSENCE DEFINITION?")
print("=" * 84)
print(f" {'absence set':<34}{'n':>6}{'model':>8}{'baseline':>10}"
      f"{'diff':>8}{'95% CI on difference':>24}")
print(" " + "-" * 82)
out_rows = []
for name, dat in variants.items():
    d = zscore(dat, dat, ERA5_FEATURES)
    d = d[d.Week.isin(TEST_WEEKS)].dropna(subset=ERA5_FEATURES)
    if d.Presence.nunique() < 2:
        continue
    y = d.Presence.values
    proba = model.predict_proba(d[ERA5_FEATURES])[:, 1]
    dist = np.array([haversine_km(r.Y, r.X, known.Y.values, known.X.values).min()
                     for _, r in d.iterrows()])
    a_m, a_b = roc_auc_score(y, proba), roc_auc_score(y, -dist)
    diff, lo, hi = paired_bootstrap(y, proba, -dist)
    print(f" {name:<34}{len(d):>6}{a_m:>8.3f}{a_b:>10.3f}{a_m - a_b:>+8.3f}"
          f"{f'[{lo:+.3f}, {hi:+.3f}]':>24}")
    out_rows.append({'absence_set': name, 'n': int(len(d)),
                     'positives': int(y.sum()), 'auc_model': round(float(a_m), 4),
                     'auc_baseline': round(float(a_b), 4),
                     'difference': round(diff, 4),
                     'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})

res = pd.DataFrame(out_rows)
res.to_csv(os.path.join(RESULTS_DIR, "tgb_absence_check.csv"), index=False)

print("\n" + "=" * 84)
print(" VERDICT")
print("=" * 84)
if len(res) == 2:
    d0, d1 = res.difference.iloc[0], res.difference.iloc[1]
    same = (d0 < 0) == (d1 < 0)
    print(f"  synthetic background   model - baseline = {d0:+.3f}")
    print(f"  target-group (TGB)     model - baseline = {d1:+.3f}")
    if same:
        print("\n  The ranking is unchanged under a survey-effort-corrected absence")
        print("  set, so the central claim is not an artefact of pseudo-absence")
        print("  sampling (Phillips et al. 2009).")
    else:
        print("\n  The ranking reverses under TGB: the primary result depends on")
        print("  the absence definition.")
    print("\n  Absolute AUCs fall under TGB by construction: surveyors go where")
    print("  they expect locusts, so TGB negatives sit in genuinely similar")
    print("  habitat.")

print("\n[SUCCESS] Saved tgb_absence_check.csv")
