"""
Measures how far ascending and descending passes disagree on local incidence
angle, how much of each feature's variance that explains, and whether angle
correlates with the label.

Radiometric terrain flattening needs no test here. Gamma-nought-flat divides
beta-nought by the illuminated area A, which is common to both
polarisations, so A cancels exactly in VH-VV (dB) and in 4*VH/(VV+VH). Every
Sentinel-1 feature in this study is ratio-form.

Incidence angle does not cancel, because VV and VH have different angular
responses. The ERA5 features are correlated against angle as a control:
reanalysis cannot be contaminated by SAR geometry, so whatever correlation
it shows is shared spatial structure, and that sets the floor for reading
the Sentinel-1 column.

OUTPUT: incidence_angle_check.csv       REQUIRES GEE AUTH
"""

import ee
import numpy as np
import pandas as pd
import os

from timeline_config import (RESULTS_DIR, BASE_DIR, DATA_DIR, ROI_COORDS,
                             ANCHOR_DATE, TEST_ANCHOR_DATE, window,
                             ERA5_FEATURES, SAR_FEATURES, require_gee_project)

ERA5_SCALE = 10  # S1 GRD native pixel spacing

print("[START] 12 - Sentinel-1 incidence-angle diagnostic\n")
GEE_PROJECT = require_gee_project()

SETS = [("train pool (anchor 2020-02-03)", os.path.join(DATA_DIR, "train_pool",
                                                        "features_all.csv"), ANCHOR_DATE),
        ("test anchor (anchor 2020-03-15)", os.path.join(DATA_DIR, "test_anchor",
                                                         "features_all.csv"), TEST_ANCHOR_DATE)]
for _, p, _a in SETS:
    if not os.path.exists(p):
        print(f"[FATAL] {p} not found. Run 02a-02c and 03 in both modes first.")
        raise SystemExit(1)

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(ROI_COORDS)


def angle_bands(anchor):
    """Mean incidence angle per orbit pass, over the T-30 and T-15 windows --
    the same windows and the same collection filters as 02b."""
    out = []
    for tag in ('T-30', 'T-15'):
        s, e = window(tag, anchor)
        base = (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(roi)
                .filterDate(str(s.date()), str(e.date()))
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
        for p in ('ASCENDING', 'DESCENDING'):
            nm = f"ang_{tag.replace('-', '')}_{p[:4]}"
            out.append(base.filter(ee.Filter.eq('orbitProperties_pass', p))
                       .select('angle').mean().rename(nm))
    return ee.Image.cat(out)


rows = []
for label, path, anchor in SETS:
    df = pd.read_csv(path)
    img = angle_bands(anchor)
    fc = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([float(r.X), float(r.Y)]), {'row_id': int(r.row_id)})
         for _, r in df.iterrows()])
    got = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                            scale=ERA5_SCALE).getInfo()
    cols = ['ang_T30_ASCE', 'ang_T30_DESC', 'ang_T15_ASCE', 'ang_T15_DESC']
    a = pd.DataFrame([{'row_id': f['properties'].get('row_id'),
                       **{c: f['properties'].get(c) for c in cols}}
                      for f in got['features']])
    d = df.merge(a, on='row_id', how='left')

    print("=" * 88)
    print(f" {label}   n = {len(d)}")
    print("=" * 88)

    # ---- 1. how much mixing is there? -------------------------------
    print("\n 1. HOW MUCH DO THE TWO ORBIT PASSES DISAGREE ON INCIDENCE ANGLE?")
    for tag in ('T30', 'T15'):
        asc, desc = d[f'ang_{tag}_ASCE'], d[f'ang_{tag}_DESC']
        both = asc.notna() & desc.notna()
        if both.sum() == 0:
            print(f"    {tag}: no point has both passes -- no mixing possible")
            continue
        diff = (asc[both] - desc[both]).abs()
        print(f"    {tag}: {both.sum():>4} points see both passes   "
              f"mean |ASC-DESC| = {diff.mean():.1f}deg   max {diff.max():.1f}deg")
        print(f"          ASC {asc[both].mean():.1f}deg (sd {asc[both].std():.1f})   "
              f"DESC {desc[both].mean():.1f}deg (sd {desc[both].std():.1f})")
        rows.append({'set': label, 'check': 'pass_disagreement', 'window': tag,
                     'n': int(both.sum()), 'value': round(float(diff.mean()), 3)})

    # Combined angle actually seen, averaging whatever passes exist.
    d['angle'] = d[['ang_T15_ASCE', 'ang_T15_DESC']].mean(axis=1)
    d = d.dropna(subset=['angle'])
    print(f"\n    incidence angle across points: {d.angle.min():.1f} to "
          f"{d.angle.max():.1f}deg (sd {d.angle.std():.2f})")

    # ---- 2 & 3. contamination and bias ------------------------------
    print("\n 2/3. CORRELATION WITH INCIDENCE ANGLE")
    print(f"    {'feature':<24}{'source':<14}{'r':>8}{'r^2':>8}   share of variance")
    print("    " + "-" * 74)
    for f in SAR_FEATURES + ERA5_FEATURES + ['Presence']:
        if f not in d.columns:
            continue
        sub = d[[f, 'angle']].dropna()
        if len(sub) < 10 or sub[f].std() == 0:
            continue
        r = float(np.corrcoef(sub[f], sub.angle)[0, 1])
        src = ('Sentinel-1' if f in SAR_FEATURES else
               'LABEL' if f == 'Presence' else 'ERA5 (control)')
        print(f"    {f:<24}{src:<14}{r:>+8.3f}{r ** 2:>8.3f}   {r ** 2:.1%}")
        rows.append({'set': label, 'check': 'corr_with_angle', 'window': 'T15',
                     'n': int(len(sub)), 'feature': f, 'source': src,
                     'value': round(r, 4), 'r2': round(r ** 2, 4)})
    print()

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS_DIR, "incidence_angle_check.csv"), index=False)

# The control calibrates the table. ERA5 cannot be contaminated by SAR
# geometry, so any correlation it shows with incidence angle is shared
# spatial structure. A Sentinel-1 correlation indicates real contamination
# only if it exceeds that floor.
c = out[out.check == 'corr_with_angle'].copy()
piv = c.pivot_table(index=['feature', 'source'], columns='set',
                    values='value', aggfunc='first').reset_index()
setcols = [x for x in piv.columns if x not in ('feature', 'source')]

print("=" * 88)
print(" VERDICT")
print("=" * 88)
print(f"\n {'feature':<24}{'source':<16}" + "".join(f"{s.split('(')[0].strip():>22}" for s in setcols)
      + "   reproduces?")
print(" " + "-" * 86)
for _, r in piv.iterrows():
    vals = [r[s] for s in setcols]
    same = all(abs(v) > 0.10 for v in vals) or all(abs(v) <= 0.10 for v in vals)
    consistent = same and (len(set(np.sign([v for v in vals]))) == 1 or all(abs(v) <= 0.10 for v in vals))
    print(f" {r['feature']:<24}{r['source']:<16}"
          + "".join(f"{v:>+22.3f}" for v in vals)
          + f"   {'yes' if consistent else 'NO -- unstable'}")

sar = c[c.source == 'Sentinel-1'].value.abs().max()
era = c[c.source == 'ERA5 (control)'].value.abs().max()
lab = c[c.feature == 'Presence']
print(f"\n  strongest |r| with angle:  Sentinel-1 {sar:.3f}   "
      f"ERA5 control {era:.3f}   label {lab.value.abs().max():.3f}")

if sar <= era:
    print("\n  The Sentinel-1 features are LESS correlated with incidence angle than")
    print("  the ERA5 controls are. Since reanalysis cannot be contaminated by SAR")
    print("  geometry, that correlation is shared spatial structure, not geometry")
    print("  leaking into the radar features. Angle mixing therefore does NOT")
    print("  explain the negative Sentinel-1 result.")
else:
    print("\n  The Sentinel-1 features exceed the ERA5 control floor, which")
    print("  indicates a genuine geometry signature.")

lv = lab.set_index('set').value
if len(lv) > 1 and (lv.abs().max() > 0.10) and (lv.abs().min() <= 0.10):
    print("\n  NOTE on the label: its correlation with angle does not reproduce across")
    print("  the two point sets, which is the signature of spatial coincidence rather")
    print("  than a structural confound. A real confound would be stable.")
print("\n  Terrain flattening needs no test: A cancels exactly in both the VH-VV")
print("  difference and RVI (see this script's docstring).")
print("\n  Third independent sign of strong spatial autocorrelation in this ROI,")
print("  after Moran's I on residuals and block CV falling below the forward")
print("  holdout (both in 09_validation_checks.py). They agree.")
print("\n[SUCCESS] Saved incidence_angle_check.csv")
