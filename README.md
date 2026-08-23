# Forecasting Observed Desert Locust Swarm Occurrence from Weather and Radar

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Reproducible without GEE](https://img.shields.io/badge/reproducible-without%20Earth%20Engine-brightgreen.svg)](#2-reproduce-the-published-results-no-earth-engine-account-needed)

**Can satellite data predict where adult locust swarms will appear, far enough ahead to be
useful?** This repository answers that for the 2020 East Africa outbreak, produces weekly risk
maps, and — deliberately — reports where the answer is *no*.

> ### Headline result
>
> ERA5 weather variables forecast swarm locations at **AUC 0.755 / recall 70.8%** with 16–37 days
> of lead time. Adding **the tested Sentinel-1 proxy features makes the model worse** (0.710). And a rule using **no
> satellite data at all** — rank locations by distance to the nearest known hopper/band site —
> beats both at **0.820**.
>
> Its only plausible role would be covering ground that field reporting cannot reach, but this
> experiment did not demonstrate operational value there. Every number here is reproduced by the
> code in this repository.

---

## Contents

- [What this repository is](#what-this-repository-is)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Reproducibility and version sensitivity](#reproducibility-and-version-sensitivity)
- [Study design](#study-design)
- [Features](#features)
- [Pipeline](#pipeline)
- [Results](#results)
- [Validation diagnostics](#validation-diagnostics)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Data provenance](#data-provenance)
- [Citation](#citation) · [License](#license)

---

## What this repository is

A complete, self-contained hindcast pipeline: raw FAO field reports go in, weekly forecast scores
and GeoTIFF risk maps come out. It is **not** an operational forecasting service and not a live
tool — it is a reproducible study of whether the sensors people assume are useful actually are.

Three things make it unusual for this problem area:

1. **Cross-life-stage design.** Trained on flightless hopper bands, tested on mobile adult swarms.
   A model trained and tested on the same immobile life stage largely re-finds the same insects
   and looks skilful without forecasting anything.
2. **A no-satellite baseline that the model has to beat** — and does not. Most published locust
   models never report one.
3. **Three validation schemes reported together** (random k-fold, spatial block, forward temporal)
   rather than the best one. The gap between them is the finding.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/siddhantjacob/swarm-landing-forecast.git
cd swarm-landing-forecast

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt          # supported current dependencies
# or, to pin the model-sensitive numerical packages used for the reference results:
# pip install -r requirements-lock.txt
```

### 2. Reproduce the published results — no Earth Engine account needed

Every extracted feature file is committed to `Data/`, so the modelling half of the pipeline runs
straight from a fresh clone:

```bash
python 04_train_and_test.py                 # weekly scores, SHAP, distance baseline, bootstrap CI
python 04_train_and_test.py --with-smap     # the ERA5 + SMAP configuration
python 04_train_and_test.py --target presence
python 06_generate_risk_map.py              # two-pass percentile risk map from the shipped rasters
python 07_validate_risk_map.py              # map-level recall and operational cost
python 08_make_figures.py                   # figures, QGIS styles, overlays
python 09_validation_checks.py              # the six validation diagnostics
```

Output lands in `results/`. The copies already committed there are the reference run — after your
own run, `git diff results/` shows exactly what your environment changed. Numerical tables should
match under the reference numerical versions; plotting-library versions can change PNG rendering.
See [version sensitivity](#reproducibility-and-version-sensitivity) and
[`results/README.md`](results/README.md).

### 3. Full rebuild from the raw FAO reports

Step `01` runs offline. Steps `02a–d`, `05`, `10`, `11` and `12` reach out to Google Earth Engine
and need an authenticated session:

```bash
earthengine authenticate
export GEE_PROJECT_ID=your-gee-cloud-project
# Windows PowerShell: $env:GEE_PROJECT_ID = "your-gee-cloud-project"
```

Then, in numerical order:

```bash
python 01_prepare_points.py

for MODE in "--set train" "--anchor test" "--set presence"; do
    python 02a_extract_era5.py         $MODE
    python 02b_extract_sar_moisture.py $MODE
    python 02c_extract_sar_rvi.py      $MODE
    python 02d_extract_smap.py         $MODE      # train and test only
    python 03_merge_features.py        $MODE
done

python 04_train_and_test.py
python 04_train_and_test.py --with-smap
python 04_train_and_test.py --target presence

python 05_export_risk_rasters.py       # exports to Google Drive -- see note below
python 06_generate_risk_map.py
python 07_validate_risk_map.py
python 08_make_figures.py

python 09_validation_checks.py
python 10_plot_wind.py
python 11_tgb_absence_check.py
python 12_incidence_angle_check.py
```

**Scripts run in numerical order.** Every script depends only on lower-numbered ones, so `01` →
`12` is a valid execution order with no exceptions.

`05` is the only step needing manual intervention: it writes five GeoTIFFs to the
`LocustRiskRasters` folder in your Google Drive (override it with `GEE_DRIVE_FOLDER`) and you
download them into `Data/risk_rasters/` yourself, keeping the exported filenames
(`Precip_T45.tif`, `Precip.tif`, `Temp.tif`, `SoilTemp_T30.tif`, `SoilTemp_T15.tif`). Those five
files are already committed here, so you only need this if you are rebuilding from scratch. Since
`09`–`12` depend only on `03`, they can be run while that export completes.

Every script checks its inputs on startup and fails with a specific next step if something is
missing.

---

## Repository layout

| Path | What it holds |
|---|---|
| `01`–`12` `*.py` | the pipeline, in execution order |
| `timeline_config.py` | single source of truth for every date, window, split and feature list |
| `source_data/` | **inputs** — unmodified FAO Data Catalogue field reports, plus Bulletin 497 |
| `Wave1_EastAfrica_Copernicus_Habitat_Mask_100m.tif` | **input** — habitat mask the absence sampler draws from |
| `Data/` | point sets and extracted feature CSVs, plus the exported risk rasters |
| `results/` | every table, figure and CSV the scripts produce — committed as a reference run |
| `docs/METHODOLOGY.md` | full methodology write-up, for independent review |
| `somaliland_test/` | completed exploratory out-of-region check (see its own README) |

`timeline_config.py` locates everything relative to itself, so the clone works from any directory.
Nothing outside this folder is imported.

---

## Reproducibility and version sensitivity

`RandomForestClassifier(random_state=42)` is deterministic **within** a scikit-learn version and
only within one. The same scripts on the same input CSVs, on two machines:

| Metric | sklearn 1.7.2 (Py 3.10) | sklearn 1.9.0 (Py 3.14) |
|---|---|---|
| Random 5-fold CV AUC | 0.897 | 0.897 |
| Forward holdout AUC | 0.752 | **0.764** |
| Optimism gap | +0.145 | +0.133 |
| Spatial block CV AUC | 0.681 | **0.718** |
| Block CV recall | 48.8% | 52.0% |
| Forward holdout recall | 73.2% | 71.5% |
| Moran's I, training residuals | 0.0982 | 0.1049 |

**What survives:** the ordering of the three validation schemes, block CV sitting below the forward
holdout, the sign of every comparison, and every stated finding. **What does not:** the third
decimal place.

> **Working rule, which every result below already satisfies: never treat an AUC difference smaller
> than about 0.02 as meaningful.** The headline model-vs-baseline gap is 0.062 with a bootstrap CI
> excluding zero; the Sentinel-1 degradation is 0.045. Both clear it.

**Environment of record** — the reference numerical results came from Python 3.14.3,
scikit-learn 1.9.0 and numpy 2.4.2. `requirements-lock.txt` pins the two model-sensitive
packages; it is deliberately not a full environment lock. Plot bytes also depend on SHAP,
Matplotlib and their rendering stack.

### What was actually verified

The complete offline workflow was run in an isolated copy on Python 3.14.3 with scikit-learn 1.9.0
and numpy 2.4.2. It reproduced the reported numerical results: weekly AUCs
0.713 / 0.765 / 0.774 / 0.769; mean 0.7554; baseline 0.8205; pooled difference −0.0622 with CI
[−0.0984, −0.0262]; the four corrected feature-set configurations
(0.755 / 0.710 / 0.767 / 0.736); random 5-fold 0.897, forward holdout 0.764, block CV 0.718;
optimism gap +0.133; 55 and 159 distinct ERA5 cells; and the full risk-map table —
272 / 6 / 1 / 15 swarms by class, 84.8% alert recall, +10,002 km² for 7 more swarms.

The result CSVs and GeoTIFFs were byte-identical to the committed references. The three SHAP PNGs
were numerically equivalent but not byte-identical: 3.8–4.2% of pixels changed under a different
SHAP/Matplotlib rendering stack. Regenerating `01` also serialised two point CSVs with ISO dates and
slightly more coordinate digits (maximum difference below 5×10⁻⁹ degrees), without changing rows,
labels or downstream results. PNG bytes and harmless CSV formatting are therefore not part of the
reproducibility contract; reported values and scientific comparisons are.

Every stochastic computational step is explicitly seeded: the forest (`random_state=42`), the
2,000-resample bootstrap, background sampling (`seed=100/200/300+wk`), stratified k-fold shuffling
and target-group sampling. The k-center base placement in `07` is deterministic. SHAP beeswarm
jitter is cosmetic and does not enter any reported metric.

**Cross-version drift.** The same clone on scikit-learn 1.8.0 gave mean model AUC 0.747 against an
*identical* baseline of 0.820, pooled difference −0.075, CI still excluding zero, logistic
regression identical at 0.624. That is 0.008 of movement on the model — under the 0.02 rule above —
and the deterministic distance baseline reproduces to the digit on every version tested, which is
what you would expect given it involves no learner at all.

---

## Study design

| | |
|---|---|
| Region | 36–39°E, 0–6°N — northern Kenya into southern Ethiopia, ~220,000 km² |
| Train on | hopper + band sightings, 18 Feb – 23 Mar 2020 (125 sightings) |
| Test on | adult swarm sightings, 31 Mar – 27 Apr 2020 (463), scored one week at a time |
| Lead time | 16 / 23 / 30 / 37 days |
| Model | Random Forest, 100 trees, depth 5, balanced classes, threshold 0.5, nothing tuned |

**Why train on bands and test on swarms.** Hopper bands cannot fly. A model trained and tested on
bands largely re-finds the same immobile insects and looks skilful without forecasting anything.
Swarms are mobile, so a correct prediction cannot be a re-sighting.

**What the test target represents.** Hopper and band sightings mark locations associated with
confirmed recent breeding activity. The forward target is observed adult-swarm occurrence, treated
as a proxy for potential settlement habitat; a swarm observation does not confirm landing,
egg-laying or subsequent breeding.

**The design does not assume the April swarms descend from the February–March bands.** Development
timing makes that plausible for part of them (nymphal development ~25–50 days, FAO Desert Locust
Guidelines), but this ROI is not a closed system. FAO Bulletin 497 records swarms crossing into
Kenya from Ethiopia and Somalia, and leaving westward into Uganda and South Sudan. Measured against
the training band sites, April swarms sit a **median of 14 km away, with 24% beyond 30 km** of any
of them:

| Distance to nearest band site | Swarms | Share |
|---|---|---|
| 0–2 km | 54 | 11.7% |
| 2–10 km | 133 | 28.7% |
| 10–30 km | 165 | 35.6% |
| 30–60 km | 95 | 20.5% |
| 60–120 km | 16 | 3.5% |

Locally fledged adults and immigrant swarms are both in the test set. The design requires only that
swarms are mobile relative to bands — not that they are the same insects. The result therefore
concerns **where adult swarms were observed under comparable environmental conditions**, as a proxy
for potential settlement habitat, not the trajectory of one cohort or proof of settlement or breeding.

---

## Features

Nine predictors along the breeding timeline, each measured in a window counted back from an anchor
date. Training features use a 3 February anchor, test features a 15 March one.

| Window | Features | Meaning |
|---|---|---|
| T−45 | `Precip_T45` | rainfall at the published 41–64 day lag before bands appear |
| T−45 → T−30 | `RVI_Phenology_Delta` | radar-measured greening |
| T−30 → T−15 | `Precip`, `Temp`, `SoilTemp_T30` | rainfall trigger, ground temperature |
| T−15 → T0 | `Moisture_T30`, `Moisture_T15`, `SoilTemp_T15` | soil wetness and warmth at laying |
| T0 → T+15 | `RVI_T0` | vegetation available at hatching |

Five from ERA5-Land reanalysis, four from Sentinel-1. Comparing the two sets is how this study
answers its question. No Sentinel-2: cloud cover left NDVI missing for over 90% of test sightings.

Soil temperature uses ERA5-Land level 1 (0–7 cm). Locust eggs sit nearer 5–10 cm, straddling the
level-1/level-2 boundary; level 1 responds fastest to surface forcing. The mismatch is stated, not
tested.

`COPERNICUS/S1_GRD` arrives orbit-corrected, noise-removed, calibrated to σ⁰ and terrain-corrected,
so no such step appears in the extraction scripts. This study adds IW/VV+VH filtering, a per-image
30 m speckle filter, a −30 dB mask after smoothing, the ratio or RVI formed per image before
temporal reduction, and a median composite over both orbit passes. Radiometric terrain flattening
is deliberately omitted: γ⁰-flat divides β⁰ by the illuminated area, which is common to both
polarisations and cancels exactly in `VH − VV` and `4·VH/(VV+VH)`. Details in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §3.

Features are z-scored within their own anchor, so a value means *warmer or cooler than normal for
this time of year*. The region warms ~2 K between anchors while sighting and non-sighting locations
differ by ~4 K, so without this the model reads every March point as too warm.

---

## Pipeline

| # | Script | Does | GEE |
|---|---|---|---|
| 01 | `01_prepare_points.py` | builds the three point sets | |
| 02a–d | `02a`–`02d` | ERA5, Sentinel-1 moisture, RVI, SMAP | ✓ |
| 03 | `03_merge_features.py` | joins features, asserts no leakage | |
| 04 | `04_train_and_test.py` | trains, scores each week, SHAP, baselines, bootstrap CI | |
| 05 | `05_export_risk_rasters.py` | region-wide feature rasters | ✓ |
| 06 | `06_generate_risk_map.py` | two-pass percentile risk map | |
| 07 | `07_validate_risk_map.py` | scores the map, operational cost | |
| 08 | `08_make_figures.py` | figures, QGIS styles, overlays | |
| 09 | `09_validation_checks.py` | six validation diagnostics | |
| 10 | `10_plot_wind.py` | wind-field figure (descriptive only) | ✓ |
| 11 | `11_tgb_absence_check.py` | target-group background robustness check | ✓ |
| 12 | `12_incidence_angle_check.py` | Sentinel-1 geometry diagnostic | ✓ |

---

## Results

All tables below are produced by the code in this repository. The CSVs behind them are in
`results/`.

### Weekly forecast, ERA5 only

| Week | Lead | Swarms | AUC | Precision | Recall |
|---|---|---|---|---|---|
| W7 | 16 d | 51 | 0.713 | 68.0% | 66.7% |
| W8 | 23 d | 154 | 0.765 | 73.2% | 70.8% |
| W9 | 30 d | 118 | 0.774 | 69.4% | 72.9% |
| W10 | 37 d | 140 | 0.769 | 68.5% | 72.9% |
| **Mean** | | **463** | **0.755** | **69.7%** | **70.8%** |

Performance is flat across 16–37 days; lead time is not the limiting factor in this range.

### Feature sets

| Configuration | AUC | Recall |
|---|---|---|
| ERA5 only (5) | 0.755 | 70.8% |
| ERA5 + Sentinel-1 (9) | **0.710** | 64.1% |
| ERA5 + SMAP (7) | **0.767** | 68.4% |
| ERA5 + Sentinel-1 + SMAP (11) | 0.736 | 58.2% |

**Sentinel-1 degrades the model; SMAP improves it.** Both are microwave soil-moisture information
tested identically. SMAP is a retrieved product; the Sentinel-1 feature is a backscatter proxy,
never validated against in-situ measurement. In the 11-feature model `SMAP_Surface_T15` ranks
second of eleven at 16.6%, while the three lowest are all Sentinel-1.

No operational Sentinel-1 soil moisture retrieval exists for East Africa — the Copernicus 1 km
product covers Europe only — which is why a proxy was used and why global SMAP outperformed it.

### Seasonal correction

| | AUC | Recall |
|---|---|---|
| ERA5, corrected | **0.755** | **70.8%** |
| ERA5, uncorrected | 0.614 | 41.7% |
| ERA5 + SMAP, uncorrected | **0.498** | 39.9% |

SMAP without correction falls below chance — soil moisture has a large seasonal cycle relative to
its class signal.

### The distance baseline

A model using no satellite data at all, ranking locations by distance to the nearest known
hopper/band site, scores mean AUC **0.820** against the satellite model's 0.755.

| Week | Model | Baseline | Difference | 95% CI | Verdict |
|---|---|---|---|---|---|
| W7 | 0.713 | 0.797 | −0.083 | [−0.205, +0.038] | indistinguishable |
| W8 | 0.765 | 0.849 | −0.084 | [−0.151, −0.019] | baseline wins |
| W9 | 0.774 | 0.803 | −0.029 | [−0.098, +0.045] | indistinguishable |
| W10 | 0.769 | 0.833 | −0.064 | [−0.135, +0.008] | indistinguishable |
| **Pooled (n = 943)** | | | **−0.062** | **[−0.098, −0.026]** | **baseline wins** |

Paired bootstrap, 2,000 resamples, interval on the difference. The pooled interval excludes zero;
only one of four weeks does so individually. Quote the pooled figure, and say *does not beat*
rather than *is worse than*.

**The satellite model does not beat proximity to known infestations.** Its value is as a
supplementary layer over ground proximity cannot cover, not as a replacement for field reporting.

### Model class

Random Forest **0.755**, logistic regression **0.624**, distance baseline **0.820**. The forest
beats a linear model by 0.131, so the feature responses really are non-linear — which defends the
model class and is orthogonal to the baseline comparison it still loses.

### SHAP

`results/fig_shap_summary.png` shows the model as fitted on training points and as applied to April
swarms.

| Feature | \|SHAP\| train | \|SHAP\| test | Direction | Transfer |
|---|---|---|---|---|
| `Precip_T45` | 0.1260 | 0.1137 | wetter → higher risk | 0.90× |
| `Temp` | 0.0673 | 0.0585 | warmer → higher risk | 0.87× |
| `Moisture_T15` | 0.0612 | 0.0652 | wetter → higher risk | 1.07× |
| `SoilTemp_T15` | 0.0592 | 0.0652 | cooler → higher risk | 1.10× |
| `SoilTemp_T30` | 0.0421 | 0.0402 | cooler → higher risk | 0.96× |
| `RVI_T0` | 0.0239 | 0.0262 | greener → higher risk | 1.10× |
| `Precip` | 0.0207 | 0.0114 | wetter → higher risk | 0.55× |
| `RVI_Phenology_Delta` | 0.0132 | 0.0102 | declining → higher risk | 0.77× |
| `Moisture_T30` | 0.0109 | 0.0105 | wetter → higher risk | 0.97× |

Three readings. The learned relationships are ecologically coherent — rainfall at T−45 dominates
and points the expected way. `Precip_T45` carries about half of all feature impact, so the model is
close to a rainfall-lag model with corrections. And the radar features are *uninformative rather
than non-transferring*: they have the smallest impact in both panels with transfer ratios near 1.0,
so the cross-target design did not break them.

### Presence as an alternative target

| Target | Model | Baseline | Gap |
|---|---|---|---|
| Swarms only | 0.755 | 0.820 | −0.065 |
| Presence (all categories) | 0.778 | 0.846 | −0.068 |

Presence is easier for the model and equally easier for the baseline. The conclusion is invariant
to target choice. Caveat: 463 of 649 presence sightings are swarms, so this does not strongly test
whether hoppers are easier.

### Absence definition

The sharpest criticism is that the test weeks contain no real surveyed absences. `11` re-runs the
comparison against target-group background (Phillips et al. 2009, *Ecological Applications*
19(1):181–197) drawn from the pooled footprint of locations FAO demonstrably surveyed.

Of 1,800 grid cells at 0.1°, **608 (33.8%) hold any FAO observation** — so about two thirds of
synthetic absences assert "no swarm here" for places nobody visited.

| Absence set | n | Model | Baseline | Difference | 95% CI |
|---|---|---|---|---|---|
| Synthetic background | 943 | 0.764 | 0.827 | −0.063 | [−0.099, −0.026] |
| Target-group background | 926 | 0.650 | 0.723 | −0.073 | [−0.122, −0.023] |

The ranking is unchanged and both intervals exclude zero. Absolute AUCs fall ~0.11 under TGB by
design, since surveyed negatives sit in similar habitat.

### Wind

| Week | Speed | Blowing toward | From |
|---|---|---|---|
| W1–W5 mean | 2.58 m/s | 297° WNW | 117° ESE |

Wind blew toward the west-northwest in every training week, steady to within 7°. Corroborated by
FAO Bulletin 497, which records swarms reaching DR Congo from Uganda "during strong easterly
winds". Descriptive only — wind is not a model feature, and the WNW corridor leads out of the ROI,
so this cannot be used to argue that wind explains where the swarms in this study landed.

### Risk maps

`Data/risk_rasters/risk_operational.tif` classifies each pixel as known infested area (within 30 km
of a reported band site) or HIGH / MEDIUM / LOW from the model beyond it, using the top 5 / 10 / 25%
of model score. `08_make_figures.py` writes a matching `.qml` beside each raster, so QGIS styles
them on load.

| Class | Swarms | % swarms | % area | Concentration |
|---|---|---|---|---|
| Known infested | 272 | 82.7% | 27.7% | **2.99×** |
| HIGH | 6 | 1.8% | 3.6% | 0.51× |
| MEDIUM | 1 | 0.3% | 2.4% | 0.13× |
| LOW | 15 | 4.6% | 9.3% | 0.49× |

Weekly recall inside alerted areas: 80.5% / 88.8% / 77.5% / 88.2%. **Point-level skill does not
transfer to a gridded product** — the known-infestation layer contributes 272 of 279 detections
(97.5%), and beyond it model-flagged areas hold *fewer* swarms than random selection.

Operational cost, as bases needed to service the alert within a given radius (k-center, farthest-
point clustering, Gonzalez 1985):

| Service radius | Known infested | Full alert | Extra bases | Per extra swarm |
|---|---|---|---|---|
| 25 km | 87 | 126 | **+39** | 5.6 |
| 50 km | 32 | 43 | +11 | 1.6 |
| 75 km | 18 | 21 | +3 | 0.4 |

Going from the known infested area to the full alert adds **10,002 km² and finds 7 more swarms**
(2.1%). Report this with the 84.8% alert recall, never the recall alone.

---

## Validation diagnostics

| Scheme | AUC | Controls for |
|---|---|---|
| Random 5-fold CV | **0.897** | nothing — neighbours in both folds |
| Forward temporal holdout | **0.764** | time |
| Spatial block CV (0.5°, 56 blocks) | **0.718** | space |

Optimism gap **+0.133**. Published locust models routinely report above 0.90; this model reports
0.897 under the same protocol, so the difference from the literature is the evaluation scheme, not
the model (Roberts et al. 2017, *Ecography* 40:913–929; Ploton et al. 2020, *Nature Communications*
11:4540).

The same gap in recall, on the training weeks:

| Scheme | AUC | Recall | Found |
|---|---|---|---|
| In-sample — *memorised, not a result* | 0.989 | 95.2% | 119/125 |
| Random 5-fold CV | 0.893 | 85.6% | 107/125 |
| Forward holdout | 0.764 | **71.5%** | 331/463 |
| Spatial block CV | 0.688 | **52.0%** | 65/125 |

**Block-CV recall falls below forward-holdout recall.** Removing spatial proximity costs more than
forecasting six weeks forward — the distance-baseline result from a fourth direction. Training
weeks are unevenly loaded: W5 alone holds 46% of the 125 sightings.

**Moran's I on residuals**: 0.105 training, **0.475** holdout, both p = 0.001 (binary weights within
30 km, 999 permutations). The model leaves substantial spatial structure unexplained, so the
independence assumption is violated and reported metrics are upper bounds.

**Effective sample size**: 125 training sightings occupy **55 distinct ERA5 cells**; 463 holdout
sightings occupy 159. Sightings inside a cell share an identical feature vector, so quote both.

**Real vs synthetic absences** (W4–W5, the only weeks holding both):

| Absences | n | Model | Baseline | Baseline lead |
|---|---|---|---|---|
| Real surveyed | 51 | 0.610 | 0.639 | +0.029 |
| Synthetic | 35 | 0.820 | 0.834 | +0.014 |

The baseline's lead does not shrink against real absences, answering the criticism. But both
collapse — absolute AUCs are optimistic by roughly 0.2 while the comparison is unaffected.

**Sentinel-1 geometry**: combining orbit passes mixes local incidence angle by 3.7–3.9°. The
Sentinel-1 features correlate with angle at up to 0.250, but the ERA5 controls reach 0.389 — and
reanalysis cannot be contaminated by SAR geometry, so that correlation is shared spatial structure.
Angle mixing does not explain the negative radar result.

---

## Limitations

- **One season, one region.** Effective training sample is 55 ERA5 cells, and no cross-year
  validation was possible.
- **The ROI censors a documented westward dispersal corridor.** FAO Bulletin 497 records swarms
  moving from Kenya "west to Uganda and northwest to Lake Turkana", reaching Magwi in South Sudan
  and Ituri in DR Congo. Those destinations lie west of the 36°E boundary; 230 swarm records in
  30–36°E during the study period are excluded by construction. Any statement about direction of
  travel computed from in-ROI sightings understates westward movement. The model–baseline
  comparison is unaffected, since both are scored on identical points.
- **Residual spatial autocorrelation.** Moran's I 0.475 on holdout residuals.
- **Sentinel-1 soil moisture is a backscatter proxy**, not a validated retrieval. The negative
  result may reflect proxy quality rather than a limitation of radar as such.
- **Resolution.** ERA5-Land is ~11 km. Maps are drawn at 1.1 km so the habitat mask applies, but the
  information is 11 km — a district-level product.
- **Precision does not transfer to maps.** Point-level precision comes from a roughly balanced
  design; swarms are rare across 220,000 km².
- **Survey-effort bias.** "No swarm reported" is not "no swarm landed". Every figure is an upper
  bound on apparent performance.
- **The habitat mask carries no information.** 71.5% of April swarms fall on Copernicus habitat
  against 74.9% areal coverage — no better than chance. It only restricts the search area.
- **Single model family.** Random Forest only, with a logistic regression comparison but no test of
  whether another learner changes the conclusion.

---

## Conclusion

The tested Sentinel-1 proxy features did not improve this model; they reduced its performance at
every stage tested. ERA5 weather variables, seasonally corrected and anchored close to the forecast
period, predict swarm locations at AUC ≈ 0.76 and recall ≈ 71% with 16–37 days of lead. Adding
SMAP L-band soil moisture improves on that slightly, which suggests the radar result reflects the
absence of a soil-moisture retrieval for this region rather than a limitation of microwave sensing
in principle.

A distance-only rule using no satellite data does better still, by 0.062 AUC with a bootstrap
interval excluding zero. The honest conclusion is that satellite habitat suitability is a weaker
predictor of swarm location than proximity to known infestations. Its only plausible role would be
covering ground that field reporting cannot reach, but this experiment did not demonstrate
operational value there: doing so cost 39 extra survey bases for 7 additional swarms found.

---

## Data provenance

| Source | Used for | Licence / terms |
|---|---|---|
| [FAO Data Catalogue — Desert Locusts Observations](https://data.apps.fao.org/catalog/dataset/desert-locusts-observations) | all field-report CSVs in `source_data/` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution required |
| FAO *Desert Locust Bulletin* 497 (5 March 2020) | corroborating narrative in `source_data/` | FAO publication terms |
| ERA5-Land (`ECMWF/ERA5_LAND/DAILY_AGGR`) via Earth Engine | five weather features | Copernicus / CDS |
| Sentinel-1 GRD (`COPERNICUS/S1_GRD`) via Earth Engine | four radar features | Copernicus |
| SMAP L4 v7 (`NASA/SMAP/SPL4SMGP/007`) via Earth Engine | soil-moisture comparison | NASA, public domain; historical version pinned for reproduction |
| Copernicus Global Land Cover | the habitat mask raster | Copernicus |

Files in `source_data/` are unmodified downloads. Nothing there was produced by this study.
SMAP v7 has been superseded by v8; changing the collection would constitute a new experiment, so
v7 is retained deliberately. Required acknowledgements, restrictions and disclaimers for bundled
third-party material and derived products are recorded in [`NOTICE`](NOTICE).

> **One parsing trap, documented because it caused a real error.** The four `*_2020.csv` files are
> ISO-dated; `Desert locusts observation by day (Global).csv` is `DD-MM-YYYY`. Parsing the latter
> without `dayfirst=True` silently drops 59% of in-ROI rows as `NaT` and misparses the rest —
> `03-06-2020` (3 June) reads as 3 March. `01_prepare_points.py` sets `dayfirst=True` for that file
> only.

---

## Citation

If you use this code or its findings, please cite it via [`CITATION.cff`](CITATION.cff), or:

> Jacob, S. (2026). *Forecasting Observed Desert Locust Swarm Occurrence from Weather and Radar*.
> MSc thesis code, Cranfield University.

---

## License

[MIT](LICENSE) for the code. Third-party data and derived materials are not relicensed under MIT;
see [Data provenance](#data-provenance) and [`NOTICE`](NOTICE).
