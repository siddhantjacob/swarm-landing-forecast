"""
Single source of truth for every date, window, split and feature list.

WINDOWS, counted back from an anchor date:
    T-45   pre-rain vegetation baseline (RVI)
    T-30   rainfall trigger and early green-up
    T-15   oviposition-window soil moisture
    T0     vegetation confirmation at hatching

ANCHORS:
    train  2020-02-03, labels W1-W5   hopper + band
    test   2020-03-15, labels W7-W10  swarm

Feature windows follow Google Earth Engine's half-open `filterDate` convention:
the start is inclusive and the end is exclusive. The train T0 interval
[2020-02-03, 2020-02-18) is therefore adjacent to W1, which opens on
2020-02-18, but does not overlap it. The test anchor retains a full calendar-day
gap before W7. 03_merge_features.py checks these boundaries at runtime.

NO SENTINEL-2 ANYWHERE. Cloud cover left NDVI missing for over 90% of test
positives in earlier iterations; Sentinel-1 RVI is the cloud-penetrating
substitute.

Wind is not a model feature. 10_plot_wind.py draws the field as a
descriptive figure and needs nothing from here.
"""

import os
from datetime import datetime, timedelta

# Google Earth Engine Cloud project used by 02a-02d, 05, 10, 11 and 12.
# Set without editing this file: export GEE_PROJECT_ID=your-project
PROJECT_ID = os.environ.get('GEE_PROJECT_ID')


def require_gee_project():
    """Return the caller's GEE project or stop with an actionable message."""
    if not PROJECT_ID:
        print("[FATAL] GEE_PROJECT_ID is not set.")
        print("        Set it to a Google Cloud project registered for Earth Engine.")
        raise SystemExit(1)
    return PROJECT_ID
HOTSPOT_NAME = 'Wave1_EastAfrica'
ROI_COORDS = [36.0, 0.0, 39.0, 6.0]      # [minLon, minLat, maxLon, maxLat]

ANCHOR_DATE = datetime(2020, 2, 3)

WINDOW_OFFSETS = {
    'T-45': (-45, -30),
    'T-30': (-30, -15),
    'T-15': (-15, 0),
    'T0':   (0, 15),
}

LABEL_WINDOWS = {
    'Train':   ('2020-02-18', '2020-03-16'),
    'Test_W1': ('2020-03-17', '2020-03-23'),
    'Test_W2': ('2020-03-24', '2020-03-30'),
}

TEST_PHASES = ['Test_W1', 'Test_W2']

# =====================================================================
# WEEKLY GRID -- 10 consecutive label weeks. Feature intervals use the
# half-open Earth Engine convention [start, end), while label weeks are
# represented as inclusive calendar dates.
# =====================================================================
# TRAIN: T0 is [2020-02-03, 2020-02-18); W1 starts 2020-02-18. These are
# adjacent and non-overlapping because the feature end is exclusive.
# TEST: T0 is [2020-03-15, 2020-03-30); W7 starts 2020-03-31, leaving a full
# unused calendar day (2020-03-30) between the feature interval and labels.
WEEK_ONE_START = datetime(2020, 2, 18)
N_WEEKS = 10
WEEK_STARTS = [WEEK_ONE_START + timedelta(days=7 * i) for i in range(N_WEEKS)]


def week_bounds(wk):
    """1-indexed. Returns (start, end) datetimes, inclusive, for week `wk`."""
    start = WEEK_STARTS[wk - 1]
    return start, start + timedelta(days=6)


def week_lead_days(wk):
    """Days from ANCHOR_DATE to the start of week `wk` -- the forecast lead."""
    return (WEEK_STARTS[wk - 1] - ANCHOR_DATE).days


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

# Every table, figure and CSV a script produces lands here, so a fresh clone
# keeps generated output separate from code and inputs. The copies shipped in
# the repository are the reference run described in README.md -- overwrite
# them freely, then `git diff` to see what your environment changed.
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def window(name, anchor=None):
    """Returns (inclusive_start, exclusive_end) for a feature window.

    `anchor` defaults to the TRAIN anchor. Pass TEST_ANCHOR_DATE to get the
    re-anchored test windows -- see the block below.
    """
    a = anchor or ANCHOR_DATE
    o1, o2 = WINDOW_OFFSETS[name]
    return a + timedelta(days=o1), a + timedelta(days=o2)


# =====================================================================
# SECOND ANCHOR FOR TESTING.
#
# THE PROBLEM WITH ONE ANCHOR. Every feature is frozen at 2020-02-03, so
# scoring an April week is not "a longer forecast" -- it is the SAME
# 3-month-old inputs applied further out. Lead to W7-W10 was 57-84 days,
# far outside any deployable early-warning horizon, and recall decayed
# accordingly. This section re-anchors the test features closer to the
# swarm labels instead.
#
# WHY 03-15. With Earth Engine's exclusive end date, a 03-16 anchor would
# produce [2020-03-16, 2020-03-31), adjacent to W7 but not overlapping it.
# The selected 03-15 anchor instead produces [2020-03-15, 2020-03-30),
# retaining one full unused calendar day before W7. The reference results use
# 03-15; changing the anchor would change the extracted test features.
#
#   TEST anchor 2020-03-15 feature windows:
#     T-45  2020-01-30 .. 2020-02-14
#     T-30  2020-02-14 .. 2020-02-29
#     T-15  2020-02-29 .. 2020-03-15
#     T0    2020-03-15 .. 2020-03-30      <- closes the day before W7
#
#   Test label weeks and their lead from THIS anchor:
#     W7   2020-03-31..04-06    16 days
#     W8   2020-04-07..04-13    23 days
#     W9   2020-04-14..04-20    30 days
#     W10  2020-04-21..04-27    37 days
#
# WHAT THIS DOES NOT CHANGE: the TRAINING features stay on the 2020-02-03
# anchor with hopper+band labels W1-W5. Train and test therefore use the
# same feature DEFINITIONS (T-45/T-30/T-15/T0 relative to their own
# anchor) at different absolute dates -- which is precisely what a
# deployed system does when it re-anchors before each forecast.
#
# THE RISK THIS INTRODUCES, AND IT IS A SERIOUS ONE: seasonal drift.
# The dominant features here are ABSOLUTE temperatures (per-feature
# isolation: Temp +31.6%, SoilTemp_T15 +30.8%, SoilTemp_T30 +30.0%), and
# presence/absence are separated by only ~4 K (298.1 vs 302.2 K). East
# Africa warms measurably between an early-February anchor and a
# mid-March one. If that seasonal shift approaches the class separation,
# a February-trained model sees every March-anchored point as "too warm"
# and recall collapses.
#
# 04_train_and_test.py therefore reports results both with and without
# within-anchor z-scoring.
# =====================================================================
TEST_ANCHOR_DATE = datetime(2020, 3, 15)

REANCHOR_TRAIN_WEEKS = [1, 2, 3, 4, 5]      # hopper+band, 2020-02-18..03-23
REANCHOR_TEST_WEEKS = [7, 8, 9, 10]         # swarm only, scored one week at a time


def test_window(name):
    """Feature window for the TEST anchor (2020-03-15)."""
    return window(name, TEST_ANCHOR_DATE)


def test_lead_days(wk):
    """Days from TEST_ANCHOR_DATE to the start of week `wk`."""
    return (WEEK_STARTS[wk - 1] - TEST_ANCHOR_DATE).days


TEST_ANCHOR_DIR = os.path.join(DATA_DIR, "test_anchor")
TRAIN_POOL_DIR = os.path.join(DATA_DIR, "train_pool")
PRESENCE_DIR = os.path.join(DATA_DIR, "presence_anchor")


def resolve_extraction(argv=None):
    """Point set + anchor selection for the 02x extraction scripts.

    Returns (points_path, anchor_date, out_dir, tag).

    Three combinations are needed, because the study pairs two different
    TARGETS with two different ANCHORS:

      (no flag)        swarm+adult  @ TRAIN anchor  -> Data/
                       the single-anchor experiments (04-08)

      --set train      hopper+band  @ TRAIN anchor  -> Data/train_pool/
                       the TRAINING pool for every model in this folder

      --anchor test    swarm+adult  @ TEST anchor   -> Data/test_anchor/
                       the re-anchored TEST features (11, 12)

    Each writes to its own directory, so no run can overwrite another's
    features. Environment equivalents: LOCUST_SET=train, LOCUST_ANCHOR=test.
    """
    import sys
    argv = argv if argv is not None else sys.argv[1:]

    want_train_pool = ('--set' in argv and 'train' in argv) or \
                      os.environ.get('LOCUST_SET', '').lower() == 'train'
    want_presence = ('--set' in argv and 'presence' in argv) or \
                    os.environ.get('LOCUST_SET', '').lower() == 'presence'
    want_test_anchor = ('--anchor' in argv and 'test' in argv) or \
                       os.environ.get('LOCUST_ANCHOR', '').lower() == 'test'

    if want_train_pool and want_test_anchor:
        print("[FATAL] --set train and --anchor test are mutually exclusive.")
        print("        Training features are always built at the TRAIN anchor.")
        raise SystemExit(1)

    if want_presence:
        # ALL locust categories at the TEST anchor -- the target the published
        # literature uses, so this is what makes the study comparable to it.
        os.makedirs(PRESENCE_DIR, exist_ok=True)
        return (os.path.join(DATA_DIR, "presence_points.csv"),
                TEST_ANCHOR_DATE, PRESENCE_DIR, 'PRESENCE (all categories)')
    if want_train_pool:
        os.makedirs(TRAIN_POOL_DIR, exist_ok=True)
        return (os.path.join(DATA_DIR, "train_points.csv"),
                ANCHOR_DATE, TRAIN_POOL_DIR, 'TRAIN POOL (hopper+band)')
    if want_test_anchor:
        os.makedirs(TEST_ANCHOR_DIR, exist_ok=True)
        return (os.path.join(DATA_DIR, "all_points.csv"),
                TEST_ANCHOR_DATE, TEST_ANCHOR_DIR, 'TEST ANCHOR (swarm+adult)')
    return (os.path.join(DATA_DIR, "all_points.csv"),
            ANCHOR_DATE, DATA_DIR, 'TRAIN ANCHOR (swarm+adult)')


def resolve_anchor(argv=None):
    """Backwards-compatible shim. Prefer resolve_extraction()."""
    _, a, d, t = resolve_extraction(argv)
    return a, d, t


if __name__ == '__main__':
    print(f"ANCHOR_DATE = {ANCHOR_DATE.date()}\n")
    print("Feature windows:")
    for name in WINDOW_OFFSETS:
        s, e = window(name)
        print(f"  {name:5s}: [{s.date()}, {e.date()})")
    print("\nLabel windows (non-overlapping with the half-open feature windows):")
    for phase, (s, e) in LABEL_WINDOWS.items():
        lead_lo = (datetime.strptime(s, '%Y-%m-%d') - ANCHOR_DATE).days
        lead_hi = (datetime.strptime(e, '%Y-%m-%d') - ANCHOR_DATE).days
        print(f"  {phase:8s}: {s} .. {e}   ({lead_lo}-{lead_hi} days after anchor)")
    t0_end = window('T0')[1].date()
    print(f"\nFeature/label boundary: features end {t0_end}, "
          f"labels start {LABEL_WINDOWS['Train'][0]} -- no overlap.")
    print(f"DATA_DIR = {DATA_DIR}")

# =====================================================================
# FEATURES -- the 9 predictors, grouped by sensor.
# The ERA5 vs ALL comparison is how this study answers its title question:
# does Sentinel-1 add anything?
# =====================================================================
ERA5_FEATURES = ['Precip_T45', 'Precip', 'Temp', 'SoilTemp_T30', 'SoilTemp_T15']
SAR_FEATURES = ['RVI_Phenology_Delta', 'Moisture_T30', 'Moisture_T15', 'RVI_T0']
ALL_FEATURES = ERA5_FEATURES + SAR_FEATURES

# SMAP L4 -- a dedicated L-band soil moisture RETRIEVAL, not a backscatter
# proxy. Optional: only available if 02d_extract_smap.py has been run.
# Adding it separates "microwave soil moisture does not help" from "the
# Sentinel-1 proxy is too noisy" -- see 02d's docstring.
SMAP_FEATURES = ['SMAP_Surface_T15', 'SMAP_Root_T15']
ERA5_SMAP_FEATURES = ERA5_FEATURES + SMAP_FEATURES
ALL_PLUS_SMAP = ALL_FEATURES + SMAP_FEATURES

CONFIGS = [("ERA5 only (5 features)", ERA5_FEATURES, "era5"),
           ("ERA5 + Sentinel-1 (9 features)", ALL_FEATURES, "all")]

# Used when 04 is run with --with-smap
CONFIGS_SMAP = [("ERA5 only (5 features)", ERA5_FEATURES, "era5"),
                ("ERA5 + Sentinel-1 (9)", ALL_FEATURES, "all"),
                ("ERA5 + SMAP (7)", ERA5_SMAP_FEATURES, "era5_smap"),
                ("ERA5 + S1 + SMAP (11)", ALL_PLUS_SMAP, "all_smap")]

THRESHOLD = 0.5          # fixed before any test data was seen; nothing is tuned

# Wind is not a model feature. 10_plot_wind.py draws the field as a
# descriptive figure and needs no constants from here.


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Used only for the distance baseline."""
    import numpy as np
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
