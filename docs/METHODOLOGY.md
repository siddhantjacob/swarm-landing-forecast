# Methodology summary

Self-contained description of the pipeline in this folder, for independent review. Every number was
produced by the code here. Please be adversarial.

Environment of record: Python 3.14.3, scikit-learn 1.9.0, numpy 2.4.2.

## 1. Research question

*Can Sentinel-1 radar be exploited to forecast desert locust swarm locations?*

Scope constraints: the 2020 outbreak, East Africa.

## 2. Study design

| | |
|---|---|
| Region | 36–39°E, 0–6°N, ~220,000 km² |
| Period | 18 Feb – 27 Apr 2020, ten consecutive 7-day windows |
| Labels | FAO Locust Hub field reports (hoppers, bands, adults, swarms, `NO LOCUST`) |
| Model | Random Forest, 100 trees, `max_depth=5`, `class_weight='balanced'`, `random_state=42` |
| Threshold | 0.5, fixed before any test data was seen |
| Tuning | none |

**Cross-target design.** Train on hopper and band sightings W1–W5 (n = 125); test on adult swarm
sightings W7–W10 (n = 463), each week scored separately.

Hopper bands are flightless and remain in place for weeks, so a model trained and tested on the same
life stage largely re-detects the same immobile insects. Testing on a mobile life stage removes that.

**The training target is confirmed breeding ground.** Hopper and band records mark places where eggs
were laid and hatched. The test therefore asks whether adult swarms occur in comparable
conditions — for an arriving swarm, a habitat-selection question rather than a tracking one.

**The design does not assume a closed cohort.** Nymphal development is ~25–50 days (FAO Desert
Locust Guidelines; Symmons & Cressman 2001), which makes local fledging plausible for part of the
April population, but the ROI is not closed. FAO Bulletin 497 (5 March 2020) records swarms crossing
into Kenya from Ethiopia and Somalia and departing westward into Uganda and South Sudan. Distance
from each April swarm to the nearest training band site:

| Distance | Swarms | Share |
|---|---|---|
| 0–2 km | 54 | 11.7% |
| 2–10 km | 133 | 28.7% |
| 10–30 km | 165 | 35.6% |
| 30–60 km | 95 | 20.5% |
| 60–120 km | 16 | 3.5% |

Median 14 km, mean 20 km, maximum 119 km; **24% lie beyond 30 km of any training band site**. The
test set therefore mixes locally fledged adults with immigrants, and the design requires only that
swarms are mobile relative to bands — not that they descend from them. The claim under test concerns
**where adult swarms are observed under comparable environmental conditions**, as a proxy for
potential settlement habitat, not the trajectory of one cohort or proof of settlement or breeding.

This also bounds the re-detection concern from the other side: the 11.7% within 2 km are the cases
where a hit could conceivably be a near-re-sighting of the same breeding site, and they are a
minority of the test set.

**Dual-anchor temporal design.** Features are measured in windows counted back from an anchor.

| | Anchor | Feature span | Labels | Lead |
|---|---|---|---|---|
| Train | 2020-02-03 | 20 Dec – 18 Feb | W1–W5 | — |
| Test | 2020-03-15 | 30 Jan – 30 Mar | W7–W10 | 16 / 23 / 30 / 37 days |

Feature windows follow Earth Engine's half-open `filterDate` convention: start inclusive, end
exclusive. The train T0 interval is therefore [2020-02-03, 2020-02-18), and W1 opening on
2020-02-18 is adjacent rather than overlapping. The test T0 interval ends exclusively on
2020-03-30 and W7 opens on 2020-03-31, leaving one unused calendar day. `03_merge_features.py`
checks both cases at runtime.

## 3. Features

| Window | Feature | Source |
|---|---|---|
| T−45 | `Precip_T45` | ERA5-Land |
| T−45 → T−30 | `RVI_Phenology_Delta` | Sentinel-1 |
| T−30 → T−15 | `Precip`, `Temp`, `SoilTemp_T30` | ERA5-Land |
| T−15 → T0 | `Moisture_T30`, `Moisture_T15`, `SoilTemp_T15` | Sentinel-1 (×2), ERA5-Land |
| T0 → T+15 | `RVI_T0` | Sentinel-1 |

Optionally `SMAP_Surface_T15`, `SMAP_Root_T15` (SMAP L4, `NASA/SMAP/SPL4SMGP/007`).

`Precip_T45` follows the published 41–64 day lag between rainfall and band appearance rather than
being selected empirically. No Sentinel-2: cloud cover left NDVI missing for over 90% of test
positives in earlier iterations.

**Soil temperature depth.** ERA5-Land level 1 (0–7 cm); the layers are 0–7, 7–28, 28–100 and
100–289 cm. Locust females dig to about 10 cm and the pod sits in the lower part, so eggs
concentrate near 5–10 cm, straddling the level-1/level-2 boundary. Level 1 responds fastest to
surface forcing. A level-1 versus level-2 ablation was not run.

### Sentinel-1 preprocessing provenance

The extraction scripts are short because the earlier stages are inherited.

*Applied by GEE before the data is read* (`COPERNICUS/S1_GRD`): orbit file application, GRD border-
and thermal-noise removal, radiometric calibration to σ⁰, Range-Doppler terrain correction against
SRTM 30 m. No separate terrain-correction step appears in this study's code because geocoded,
calibrated σ⁰ is the starting point.

*Applied here*: IW mode with VV and VH; a 30 m focal-mean speckle filter per image; a residual
−30 dB mask applied after smoothing, so the threshold tests denoised pixels; the ratio (VH − VV in
dB) or RVI (`4·VH_lin/(VV_lin+VH_lin)`, computed in linear power since ratios of decibels are not
physically meaningful) formed per image before temporal reduction; then a temporal median.

*Both orbit passes combined.* This choice was inherited from an earlier ablation in the same ROI,
but with a different training anchor and point set. Removing its single-orbit lock reduced
full-training-set SAR missingness from 217/514 (42.2%) to 12/514 (2.3%); the 2.3% figure therefore
does not describe this experiment's test set. Here, moisture was complete for all 1,362 points at
the test anchor. One point lacked RVI, but it belonged to W3; the evaluated W7–W10 set had no
missing Sentinel-1 feature values. The coverage gap was an acquisition-plan issue, not a processing
fault.

**Radiometric terrain flattening is a no-op here, not an omission.** γ⁰-flat divides β⁰ by the
simulated illuminated area *A* (Vollrath et al. 2020, *Remote Sensing* 12(11):1867). *A* is
identical for both polarisations, so

```
γ⁰_VH(dB) − γ⁰_VV(dB) = (β⁰_VH − A) − (β⁰_VV − A) = β⁰_VH − β⁰_VV
4·(VH/A) / ((VV/A) + (VH/A)) = 4·VH / (VV + VH)
```

*A* cancels exactly in both. All four Sentinel-1 features are ratio-form, so slope correction would
return identical values. Ratio indices are constructed precisely to be robust to geometry, which is
why cross-pol/co-pol ratios and RVI are standard.

**Incidence-angle normalisation does not fully cancel**, because VV and VH have different angular
responses, so it is measured (`12_incidence_angle_check.py`). Combining passes mixes local
incidence angle by 3.7–3.9° on average across a 30.8–45.6° range; ascending and descending means
are nearly identical (39.4° vs 38.7°), so it adds scatter rather than bias.

| Feature | Source | Train | Test | Reproduces |
|---|---|---|---|---|
| `RVI_T0` | Sentinel-1 | +0.250 | +0.199 | yes |
| `Moisture_T15` | Sentinel-1 | +0.239 | +0.201 | yes |
| `Moisture_T30` | Sentinel-1 | +0.216 | +0.210 | yes |
| `RVI_Phenology_Delta` | Sentinel-1 | +0.075 | +0.040 | yes |
| `SoilTemp_T30` | **ERA5 control** | **−0.389** | −0.023 | no |
| `SoilTemp_T15` | **ERA5 control** | **−0.358** | −0.053 | no |
| `Temp` | ERA5 control | −0.204 | +0.040 | no |
| `Precip` | ERA5 control | −0.131 | −0.136 | yes |
| `Precip_T45` | ERA5 control | −0.026 | −0.020 | yes |
| `Presence` | **LABEL** | +0.168 | +0.040 | no |

Read the control, not the absolute numbers. The strongest correlation with SAR incidence angle
belongs to an ERA5 reanalysis variable, which cannot be contaminated by SAR geometry — so it is
shared spatial structure. Since the largest Sentinel-1 value (0.250) sits below that floor (0.389),
angle mixing does not explain the negative Sentinel-1 result. The label's correlation does not
reproduce across sets (0.168 → 0.040), the signature of spatial coincidence rather than a confound.

**What would actually be required.** σ⁰ is not soil moisture regardless of correction. Converting
it needs a physical inversion or the TU Wien change-detection approach, which establishes per-pixel
dry and wet references from the multi-year archive and expresses relative surface soil moisture as
`(σ⁰ − σ⁰_dry)/(σ⁰_wet − σ⁰_dry)`. That algorithm underpins the operational Copernicus Global Land
1 km Surface Soil Moisture product, which covers continental Europe only. **No operational
Sentinel-1 soil moisture retrieval exists for East Africa.** §4.2 is therefore a global L-band
retrieval against an uncalibrated backscatter proxy, not well-processed against badly-processed
data. Applying TU Wien to the East African archive is the substantive future work.

### Absences

Real FAO `NO LOCUST` survey records are used first; habitat-masked random background tops up any
shortfall against that week's sighting count. Realised: 51 real absences, 630 synthetic across the
test point file. The `Source` column preserves which is which.

### Seasonal correction

Each feature is z-scored against the distribution of its own anchor's point set:
`z = (x − mean(all points at that anchor)) / sd(same)`.

`SoilTemp_T15` mean is 301.07 K at the February anchor and 303.17 K at the March anchor — a +2.09 K
shift, against a ~1.9 K difference between sighting and non-sighting locations. The seasonal change
exceeds the class signal, so a February-trained model reads every March location as too warm. The
transform is label-free: mean and SD are taken over all points at that anchor.

Random Forests are scale-invariant within a dataset, so this has no effect on a conventional split.
It matters here only because train and test are standardised against different reference
distributions.

## 4. Results

### 4.1 Weekly forecast, ERA5 only, corrected

| Week | Lead | Swarms | AUC | Precision | Recall |
|---|---|---|---|---|---|
| W7 | 16 d | 51 | 0.713 | 68.0% | 66.7% |
| W8 | 23 d | 154 | 0.765 | 73.2% | 70.8% |
| W9 | 30 d | 118 | 0.774 | 69.4% | 72.9% |
| W10 | 37 d | 140 | 0.769 | 68.5% | 72.9% |
| **Mean** | | **463** | **0.755** | **69.7%** | **70.8%** |

### 4.2 Feature-set comparison

| Configuration | AUC | Precision | Recall |
|---|---|---|---|
| ERA5 only (5) | 0.755 | 69.7% | 70.8% |
| ERA5 + Sentinel-1 (9) | **0.710** | 68.0% | 64.1% |
| ERA5 + SMAP (7) | **0.767** | 70.0% | 68.4% |
| ERA5 + S1 + SMAP (11) | 0.736 | 67.8% | 58.2% |

Feature importance, 11-feature model: `Precip_T45` 18.1%, **`SMAP_Surface_T15` 16.6% (2nd)**,
`Temp` 10.2%, `SoilTemp_T30` 10.0%, `Moisture_T15` 9.7% (best Sentinel-1), `SoilTemp_T15` 9.6%,
`Precip` 6.7%, `SMAP_Root_T15` 6.6%, `RVI_T0` 4.9%, `RVI_Phenology_Delta` 4.2%, `Moisture_T30` 3.4%.

### 4.2b SHAP

Mean |SHAP|, 9-feature model, as fitted on training points and as applied to test points:

| Feature | Train | Test | Direction | Transfer |
|---|---|---|---|---|
| `Precip_T45` | 0.1260 | 0.1137 | wetter → higher risk | 0.90× |
| `Temp` | 0.0673 | 0.0585 | warmer → higher risk | 0.87× |
| `Moisture_T15` | 0.0612 | 0.0652 | wetter → higher risk | 1.07× |
| `SoilTemp_T15` | 0.0592 | 0.0652 | cooler → higher risk | 1.10× |
| `SoilTemp_T30` | 0.0421 | 0.0402 | cooler → higher risk | 0.96× |
| `RVI_T0` | 0.0239 | 0.0262 | greener → higher risk | 1.10× |
| `Precip` | 0.0207 | 0.0114 | wetter → higher risk | **0.55×** |
| `RVI_Phenology_Delta` | 0.0132 | 0.0102 | declining → higher risk | 0.77× |
| `Moisture_T30` | 0.0109 | 0.0105 | wetter → higher risk | 0.97× |

Three claims importance alone cannot support. The learned relationships are ecologically coherent
rather than sign-inverted artefacts — rainfall at T−45 dominates and points the expected way.
`Precip_T45` carries about as much impact as the next three features combined (Gini 25.1%, more
than the next two together), so the model is close to a rainfall-lag model with corrections. And
the Sentinel-1 features are uninformative rather than non-transferable: smallest impact in both
panels, transfer ratios 0.97× and 0.77×, so the cross-target design did not break them.

Mean |SHAP| measures contribution to the model's output, not to accuracy — a feature can have large
consistent SHAP values and still be wrong. Reported alongside held-out AUCs, never in place of them.

The two soil-temperature features point opposite to air `Temp`. After z-scoring these are anomalies
rather than absolute values, and cooler-than-normal soil under warmer-than-normal air is a plausible
wet-soil signature — but this is a post-hoc reading, not a hypothesis the design tested.

### 4.3 Seasonal correction effect

| Configuration | With | Without |
|---|---|---|
| ERA5 only | 0.755 | 0.614 |
| ERA5 + Sentinel-1 | 0.710 | 0.591 |
| ERA5 + SMAP | 0.767 | **0.498** |

SMAP without correction falls below chance.

### 4.4 Distance baseline

Rank test locations by great-circle distance to the nearest training hopper/band sighting; no
satellite data.

| Week | Baseline AUC |
|---|---|
| W7 | 0.797 |
| W8 | 0.849 |
| W9 | 0.803 |
| W10 | 0.833 |
| **Mean** | **0.820** |

**The baseline outperforms every satellite configuration**, including the best (0.767).

Paired bootstrap (2,000 resamples; points resampled once, both scorers re-evaluated on the same
resample, so the interval is on the difference):

| Week | Model | Baseline | Difference | 95% CI | Verdict |
|---|---|---|---|---|---|
| W7 | 0.713 | 0.797 | −0.083 | [−0.205, +0.038] | indistinguishable |
| W8 | 0.765 | 0.849 | −0.084 | [−0.151, −0.019] | baseline wins |
| W9 | 0.774 | 0.803 | −0.029 | [−0.098, +0.045] | indistinguishable |
| W10 | 0.769 | 0.833 | −0.064 | [−0.135, +0.008] | indistinguishable |
| **Pooled (n = 943)** | | | **−0.062** | **[−0.098, −0.026]** | **baseline wins** |

Only one of four weeks excludes zero individually — stated plainly, because at 51–154 sightings per
week over far fewer independent cells the interval is wide. On the presence target the pooled result
is −0.075, 95% CI [−0.102, −0.047], n = 1,298.

**Model class.** Random Forest 0.755, logistic regression 0.624, distance baseline 0.820. The
forest beats a linear model by 0.131, so the non-linearity does real work — orthogonal to, and not
a defence against, the baseline comparison.

### 4.4a Absence definition — target-group background

`11_tgb_absence_check.py` re-runs the comparison against target-group background (Phillips et al.
2009, *Ecological Applications* 19(1):181–197; applied to locusts by Yusuf et al.), drawn from the
pooled footprint of locations FAO demonstrably surveyed, so both classes share a survey-effort
footprint.

Motivating diagnostic: of 1,800 cells at 0.1°, 608 (33.8%) hold any FAO observation, so roughly two
thirds of synthetic absences assert "no swarm here" for locations nobody visited.

| Absence set | n | Model | Baseline | Difference | 95% CI |
|---|---|---|---|---|---|
| Synthetic background | 943 | 0.764 | 0.827 | −0.063 | [−0.099, −0.026] |
| Target-group background | 926 | **0.650** | **0.723** | **−0.073** | **[−0.122, −0.023]** |

Ranking unchanged, both intervals exclude zero. Absolute AUCs fall ~0.11 by design. (These pool all
four weeks into one curve; §4.1 averages four per-week AUCs.)

**Implementation note.** Week-matching the background to non-swarm survey records yields, in
W7–W10, a pool with no `NO LOCUST` records at all — only hopper/band/adult sightings, the exact
locations the distance baseline measures from. The baseline then scores 0.354 and the central claim
appears to reverse. Phillips et al. do not week-match; the pooled footprint is 57% `NO LOCUST`
records and represents survey effort.

A cell holding a swarm in a different week remains eligible as background for this week, since the
question is whether a swarm is present in week *w*. Excluding every cell that ever held a swarm
would strip out the good habitat that makes the test hard.

### 4.4b Wind — descriptive only

Wind is not a feature of any model here. `10_plot_wind.py` draws the ERA5-Land 10 m field over each
training week. Measured: 2.58 m/s blowing toward 297° (WNW), from 117° (ESE), steady to within 7°
across all five weeks. FAO Bulletin 497 corroborates the direction independently, recording swarms
reaching DR Congo from Uganda "during strong easterly winds". It must not be used to argue that
wind explains where the swarms in this study landed — the WNW corridor leads out of the ROI (§6).

### 4.5 Alternative target — presence

| Target | Model | Baseline | Gap |
|---|---|---|---|
| Swarms only | 0.755 | 0.820 | −0.065 |
| Presence (all categories) | 0.778 | 0.846 | −0.068 |

Presence is easier for the model (+0.023) and equally easier for the baseline (+0.026); the gap is
unchanged, so the conclusion is invariant to target choice. Caveat: 463 of 649 presence sightings
are swarms, so this does not strongly test whether hoppers are easier.

### 4.6 Validation diagnostics

| Scheme | AUC | Controls for |
|---|---|---|
| Random 5-fold CV | **0.897** | nothing |
| Forward temporal holdout | **0.764** | time |
| Spatial block CV (0.5°, 56 blocks) | **0.718** | space |

The same model reports 0.897 under the literature-standard protocol. Optimism gap +0.133. Spatial
leakage exceeds temporal: block CV falls below the forward holdout, so block CV is the conservative
figure (Roberts et al. 2017, *Ecography* 40:913–929; Ploton et al. 2020, *Nature Communications*
11:4540).

The same gap in recall, on W1–W5 (125 sightings):

| Scheme | AUC | Recall | Precision | Found |
|---|---|---|---|---|
| In-sample (memorised — *not a result*) | 0.989 | 95.2% | 96.0% | 119/125 |
| Random 5-fold CV | 0.893 | 85.6% | 81.7% | 107/125 |
| Forward holdout (W7–W10) | 0.764 | **71.5%** | 70.1% | 331/463 |
| Spatial block CV | 0.688 | **52.0%** | 66.3% | 65/125 |

Block-CV recall falls below forward-holdout recall: removing spatial proximity costs more than
forecasting six weeks forward, so the forward holdout is flattered by the April swarms occupying
much the same ground as the February–March bands. Four diagnostics now agree on this. Recall here
is pooled; §4.1 averages per-week recalls, hence 71.5% against 70.8%.

Training weeks are unevenly weighted: W5 alone carries 46% of the training sightings (13 / 21 / 12 /
21 / 58 across W1–W5), so the training period is dominated by 17–23 March, which sits closest to the
test anchor. Per-week recall figures are unstable across software versions and should not be quoted.

**Moran's I on residuals** (binary weights ≤30 km, 999 permutations): training 0.105, holdout
**0.475**, both p = 0.001. Positive and significant means the model leaves spatial structure
unexplained, the independence assumption is violated, and reported AUCs are upper bounds.

**Effective sample size**: 125 training sightings occupy **55 distinct ERA5 cells** (0.1°, ~11 km);
463 holdout sightings occupy 159.

**Real vs synthetic absences**, W4–W5, the only weeks holding both (train W1–W3, n = 46; test
n = 79):

| Absences | n | Model | Baseline | Baseline lead |
|---|---|---|---|---|
| Real surveyed | 51 | 0.610 | 0.639 | +0.029 |
| Synthetic | 35 | 0.820 | 0.834 | +0.014 |

The baseline's lead does not shrink against real absences — it is larger — so the criticism is
answered; n is small, so the direction is the reportable point, not the gap between the two. But
both collapse: absolute AUCs are optimistic by roughly 0.2 while the comparison is unaffected.
§4.4a runs the same test at n = 926 with intervals.

### 4.7 Gridded deployment

Applied to a 1.1 km grid over the full region (two-pass percentile classification: top 5% HIGH, 10%
MEDIUM, 25% LOW), with a separate layer for the 30 km buffer around known infestations.

| Class | Swarms | % swarms | % area | Concentration |
|---|---|---|---|---|
| Known infested (≤30 km) | 272 | 82.7% | 27.7% | **2.99×** |
| HIGH (model, beyond) | 6 | 1.8% | 3.6% | 0.51× |
| MEDIUM (model, beyond) | 1 | 0.3% | 2.4% | 0.13× |
| LOW (model, beyond) | 15 | 4.6% | 9.3% | 0.49× |

**Point-level skill does not transfer to a gridded product.** The known-infestation layer
contributes 272 of 279 detections (97.5%). Beyond it, model-flagged areas contain fewer swarms than
random selection (0.44×).

Buffer radius trade-off, no model: 5 km → 4,004 km², 24.6% recall, 10.3×; 10 km → 11,019 km², 41.3%,
6.3×; 15 km → 18,623 km², 58.4%, 5.2×; 30 km → 46,178 km², 82.7%, 3.0×. At equal area (~18,600 km²)
a 15 km buffer alone (58.4%) outperforms a 10 km buffer plus model HIGH (48.0%).

### 4.8 Operational cost

Fewest bases such that every alerted pixel lies within a service radius of one. The radius is not
fitted; a range is reported. What the design controls is the comparison between the known infested
area alone (model-free) and the full alert.

| Service radius | Known infested | Full alert | Extra bases | Extra per additional swarm |
|---|---|---|---|---|
| 25 km | 87 | 126 | **+39** | 5.6 |
| 50 km | 32 | 43 | +11 | 1.6 |
| 75 km | 18 | 21 | +3 | 0.4 |

Moving from the known infested area to the full alert adds 10,002 km² and finds 7 additional swarms
(2.1%). This is §4.7's 0.44× restated in the unit a control programme budgets in, and should always
be quoted with the 84.8% alert recall rather than the recall alone.

k-center, solved by farthest-point clustering (Gonzalez 1985, *Theoretical Computer Science*
38:293–306): deterministic, monotone, a 2-approximation, so counts are upper bounds within a factor
of 2 of optimal. Alerted pixels are subsampled to 8,000 (~2.4 km mean spacing), which biases counts
slightly low.

## 5. Conclusions

1. **The tested Sentinel-1 proxy features did not improve this model.** Adding the four features
   costs 0.045 AUC and 6.7 points of
   recall. Three of the four lowest-ranked features are radar, and SHAP shows they were never
   contributing rather than failing to transfer.
2. **SMAP does help, slightly** (0.767 vs 0.755), and ranks second of eleven features. Both are
   microwave soil-moisture information tested identically; the difference is that SMAP is a
   retrieved product and no equivalent Sentinel-1 retrieval exists for this region.
3. **Neither beats proximity.** The distance baseline scores 0.820, ahead by 0.062 with a bootstrap
   interval excluding zero, and the gap is unchanged under a survey-effort-corrected absence set.
4. **The evaluation scheme, not the model, explains the gap with the literature.** This model
   reports 0.897 under random CV and 0.718 under spatial block CV.
5. **Point-level skill does not survive gridding.** 97.5% of map detections come from the
   known-infestation buffer, and the model beyond it performs worse than random.

## 6. Stated limitations

- **Single season, single region.** Title-constrained. Effective training sample 55 ERA5 cells, not
  125 sightings. No cross-year validation.
- **The ROI censors a documented westward dispersal corridor.** FAO Desert Locust Bulletin 497
  (5 March 2020) records swarms moving from Kenya "west to Uganda and northwest to Lake Turkana", a
  swarm "coming from northeast Uganda" reaching Magwi, South Sudan on 17 February, and swarmlets
  from northwest Uganda reaching Ituri, DR Congo on 18 February "during strong easterly winds".
  Moroto (34.6°E), Magwi (32.3°E) and Aru (30.8°E) lie west of the 36°E boundary, and the FAO
  archive holds 230 swarm records in 30–36°E during the study period (97 Feb, 75 Mar, 58 Apr) that
  this ROI excludes by construction. Any statement about direction of travel computed from in-ROI
  sightings understates westward movement. The model–baseline comparison is unaffected: swarms
  leaving the ROI are unobserved for both scorers, and both are scored on identical points.
- **Residual spatial autocorrelation** (Moran's I 0.475, p = 0.001). Independence violated;
  reported metrics are upper bounds.
- **Survey-effort bias.** FAO "no report" ≠ "no locusts". All recall figures are upper bounds on
  apparent performance. Not correctable with this data.
- **Scale mismatch.** ERA5-Land ~11 km, SMAP 9 km, against point sightings. Maps drawn at 1.1 km are
  interpolated; the information is 9–11 km.
- **Precision does not transfer to maps.** Point-level precision comes from a roughly balanced
  design; swarms are rare across 220,000 km².
- **Habitat mask carries no information**: 71.5% of April swarms fall on Copernicus habitat against
  74.9% areal coverage. Used only to restrict the search domain.
- **Test weeks contain no real absences** — all 480 are synthetic. Measured cost: against real
  surveyed absences the model scores ~0.61 rather than ~0.82. The comparison is unaffected;
  absolute performance is optimistic by ~0.2 AUC.
- **Single model family.** Random Forest, with a logistic regression comparison, but no test of
  whether another learner changes the conclusion.
- **Results are not bit-reproducible across scikit-learn versions.** Differences of 0.01–0.04 AUC
  were measured between 1.7.2 and 1.9.0 on identical code and inputs. Every conclusion holds in
  both; no difference below ~0.02 AUC should be called meaningful.

## 7. Questions for a reviewer

1. Does the cross-target design (bands → swarms) adequately defeat re-detection, given 12% of April
   swarms fall within 2 km of a March band site?
2. Is a distance baseline that uses the same training sightings the model saw a fair comparator, or
   is it advantaged by construction?
3. Is within-anchor z-scoring defensible, given the test point set is 49% positives and its mean is
   therefore pulled toward swarm-like conditions?
4. Does the target-group background in §4.4a adequately answer the pseudo-absence criticism, or does
   its own construction (survey-effort-biased, cells with swarms in other weeks retained) introduce
   a different bias?
5. Is the pooled bootstrap interval the right summary when only one of four weeks is individually
   significant?
6. Given block-CV recall of 52%, is it defensible to report the forward-holdout figures as the
   headline result?
