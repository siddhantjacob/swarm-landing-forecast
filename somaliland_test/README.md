# Somaliland exploratory out-of-region check

This completed extension asks whether the unchanged model specification shows
useful transfer outside its training region. It refits the model on the same
Kenya/Ethiopia training pool and scores hopper+band sightings in Somaliland,
the nearest sizeable same-season East African cluster outside the main study
box, during the same 18 February–23 March 2020 label period.

This is a small exploratory check, not a definitive external validation. It
addresses the limitation identified in [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md):
a nearby East African site under a broadly similar rainfall regime had not been
tested.

## Results

| Configuration | AUC | Precision | Recall | Found |
|---|---:|---:|---:|---:|
| ERA5 only (5), seasonally corrected | 0.373 | 0.0% | 0.0% | 0/33 |
| ERA5 + Sentinel-1 (9), corrected | 0.436 | 18.2% | 6.2% | 2/32 |
| ERA5 only, uncorrected | 0.440 | 13.3% | 6.1% | 2/33 |
| ERA5 + Sentinel-1, uncorrected | 0.422 | 45.0% | 28.1% | 9/32 |

All four AUCs are below 0.5 in this sample, and the headline corrected ERA5
configuration finds none of the 33 sightings. This check therefore provides no
evidence of useful transfer to this Somaliland sample. It does not establish
that transfer is impossible in Somaliland generally: the sample, absence design
and reconstructed habitat mask limit that inference.

The distance baseline scores 0.442. Its difference from the corrected ERA5
model is −0.068 with a paired-bootstrap 95% CI of [−0.219, +0.076], so the two
are statistically indistinguishable here. All Somaliland points are 859–1,052
km from the nearest Kenya/Ethiopia training site (median 992 km), leaving little
distance contrast within this test.

## Interpretation constraints

- **Small positive sample.** There are 33 unique hopper sightings after the
  same coordinate-and-date deduplication used by `01_prepare_points.py`: 18 in
  W2 and 15 in W3. There are no band positives and no positives in W1, W4 or W5.
- **All absences are synthetic.** No FAO `NO LOCUST` records fall inside this
  box and period. The headline W7–W10 swarm test in the main study also uses
  synthetic absences; the main repository's W4–W5 diagnostic is the separate
  comparison that uses real surveyed absences. AUCs from synthetic-background
  designs describe discrimination against the sampled background, not verified
  true absence throughout the region.
- **One Sentinel-1-positive row is incomplete.** The nine-feature corrected
  configuration scores 32 rather than 33 positives, which is why its recall is
  reported as 2/32 (6.2%).
- **The habitat mask is reconstructed.** `S0_build_habitat_mask.py` creates a
  Somaliland equivalent from Copernicus Global Land Cover because the main mask
  covers only Kenya/Ethiopia. The original mask-building script and exact class
  selection are unavailable, so `S0` records a reasoned assumption rather than
  an exact reproduction. This limits direct comparison with the headline AUC.

## Reproduce the check

`S1` and `S3` run offline from the files committed here. `S0` and `S2` need an
authenticated Earth Engine session and are required only to rebuild the mask
and extracted features from their sources.

```bash
python S0_build_habitat_mask.py      # GEE: rebuild habitat mask
python S1_prepare_points.py          # offline: rebuild point set
python S2_extract_features.py        # GEE: rebuild ERA5 + Sentinel-1 features
python S3_merge_and_test.py          # offline: reproduce scores and baseline
```

Outputs are in `Data_somaliland/`. `S3` reports AUC, precision, recall,
found/missed counts and false alarms for both feature configurations, with and
without seasonal correction, plus the paired-bootstrap distance comparison.
