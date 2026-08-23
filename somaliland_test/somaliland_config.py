"""
Config for the Somaliland out-of-region test.

WHY THIS EXISTS
Section 3.1 of the Methodology argues that this study's second-region
exclusion is a scope limit, not a proven impossibility, and that a nearby
East African site sharing the training region's bimodal rainfall pattern
is untested and plausible future work -- distinct from West Africa, which
the study has an actual argument against (Rousseau & Betts, 2022).

Somaliland is the closest thing to a natural candidate: within the
2020-02-18..2020-03-23 training window used throughout this study, the
raw FAO hopper+band files (source_data/hoppers_2020.csv, bands_2020.csv)
show a tight cluster of 60 sightings (33 hopper, 27 band) at
10.0-10.7N, 43.9-46.1E -- the second-largest concentration of hopper/band
records anywhere in East Africa outside the original study box in this
exact period, after Eritrea.

This folder reuses the ORIGINAL trained model (fit on train_pool/, i.e.
the Kenya/Ethiopia hopper+band points, unchanged) and scores it against
this new Somaliland point set, using the exact same feature definitions,
z-scoring convention and evaluation code as 04_train_and_test.py.

WHAT IS DIFFERENT FROM THE MAIN PIPELINE, AND WHY
  1. New ROI (Somaliland cluster box instead of the Kenya/Ethiopia box).
  2. No pre-built habitat mask exists for this ROI. S0_build_habitat_mask.py
     reconstructs one via Copernicus Global Land Cover 100m in Earth Engine.
     The land-cover class selection there is a DOCUMENTED ASSUMPTION, not a
     confirmed match to however Wave1_EastAfrica_Copernicus_Habitat_Mask_100m.tif
     was originally built (that generation script is not in this repo).
     Check S0's docstring before trusting the synthetic absences it enables.
  3. Zero real FAO 'NO LOCUST' records fall inside this box in this window,
     so ALL absences here are synthetic. The main W7-W10 swarm test is also
     synthetic-only; the main repository's W4-W5 diagnostic separately tests
     real surveyed absences. This limitation is reported with the results.

Everything else -- anchor date, feature windows, model spec, threshold,
z-scoring rule -- is inherited unchanged from timeline_config.py so the
comparison is apples-to-apples with the main study's headline results.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 12_Swarm_Landing_Forecast/
sys.path.insert(0, BASE_DIR)

from timeline_config import (  # noqa: E402  (import after sys.path fix, deliberate)
    ANCHOR_DATE, WEEK_STARTS, week_bounds, window, require_gee_project,
    ALL_FEATURES, ERA5_FEATURES, SAR_FEATURES, CONFIGS, THRESHOLD,
    haversine_km, DATA_DIR as MAIN_DATA_DIR,
)

# Somaliland cluster box: lat 10.0-10.7N, lon 43.9-46.1E. Margin of ~0.1-0.2
# degrees added around the raw point extent (Lat 10.13-10.60, Lon 44.04-45.96)
# so points sitting exactly at the edge of the bin used to find this cluster
# are not clipped.
SOMALILAND_ROI_COORDS = [43.9, 10.0, 46.1, 10.7]   # [minLon, minLat, maxLon, maxLat]

# Training weeks only -- these are hopper/band sightings, scored the same way
# train_points.csv is: hopper+band, TRAIN anchor (2020-02-03), weeks W1-W5.
TRAIN_WEEKS = [1, 2, 3, 4, 5]

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SELF_DIR, "Data_somaliland")
os.makedirs(DATA_DIR, exist_ok=True)

MASK_PATH = os.path.join(SELF_DIR, "Somaliland_Habitat_Mask_100m.tif")

SRC_DIR = os.path.join(BASE_DIR, "source_data")
