"""
Builds the gridded risk map from the exported rasters.

Two passes: the first samples 5% of pixel scores to learn the top 5 / 10 /
25% thresholds, the second classifies every pixel. Percentiles rather than
absolute probabilities, because the model was fitted on a roughly balanced
point set and its probabilities do not transfer to a grid on which swarms
are rare.

The known-activity buffer is written as a separate class rather than blended
into the model surface, so the map cannot take credit for proximity the
model did not earn.

Grid features are z-scored against the grid's own distribution, which is not
the training distribution. The gap between the two is printed.

OUTPUT: risk_model_classes.tif, risk_known_buffer.tif,
        risk_operational.tif, risk_map_thresholds.json
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import os
import json
import warnings

from timeline_config import DATA_DIR, BASE_DIR, TEST_ANCHOR_DATE, REANCHOR_TRAIN_WEEKS
from timeline_config import ERA5_FEATURES, haversine_km

warnings.filterwarnings("ignore", category=RuntimeWarning)

RASTER_DIR = os.path.join(DATA_DIR, "risk_rasters")
MASK_PATH = os.path.join(BASE_DIR, "Wave1_EastAfrica_Copernicus_Habitat_Mask_100m.tif")
BLOCK = 512
KNOWN_BUFFER_KM = 30.0
PCTL = {'high': 95, 'medium': 90, 'low': 75}

print(f"[START] 06 - Generating weekly risk map (ERA5-only, test anchor "
      f"{TEST_ANCHOR_DATE.date()})...")

# ---------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------
missing = [f for f in ERA5_FEATURES if not os.path.exists(os.path.join(RASTER_DIR, f + ".tif"))]
if missing:
    print(f"[FATAL] Missing rasters in {RASTER_DIR}: {missing}")
    print("        Run 05_export_risk_rasters.py, then download the .tif files")
    print("        from Google Drive into that directory, keeping the names.")
    raise SystemExit(1)
if not os.path.exists(MASK_PATH):
    print(f"[FATAL] Habitat mask not found: {MASK_PATH}")
    raise SystemExit(1)

# ---------------------------------------------------------------------
# MODEL -- refit here rather than loaded, so the map can never silently
# disagree with 11/12 about which model it is showing.
# ---------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, 'train_pool', 'features_all.csv')

if not os.path.exists(TRAIN_FEATURES_PATH):
    print(f"[FATAL] Training pool not found: {TRAIN_FEATURES_PATH}")
    print("        Run 01b, then 02a/02b/02c --set train, then 03_merge_features.py --set train.")
    raise SystemExit(1)

train_all = pd.read_csv(TRAIN_FEATURES_PATH)
train_pool = train_all[train_all.Week.isin(REANCHOR_TRAIN_WEEKS)].dropna(subset=ERA5_FEATURES)

# z-score the training features against the training pool's own distribution,
# exactly as 04_train_and_test.py does.
tr_mu = train_all[ERA5_FEATURES].mean()
tr_sd = train_all[ERA5_FEATURES].std().replace(0, 1.0)
Xtr = (train_pool[ERA5_FEATURES] - tr_mu) / tr_sd

model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                               max_depth=5, random_state=42)
model.fit(Xtr, train_pool.Presence)
print(f" -> model fitted on {len(train_pool)} hopper+band points, "
      f"{int((train_pool.Presence == 1).sum())} sightings (ERA5-only, 5 features)")

# ---------------------------------------------------------------------
# LAYER 1 -- known-activity buffer, no model involved
# ---------------------------------------------------------------------
known = train_all[train_all.Week.isin(REANCHOR_TRAIN_WEEKS) & (train_all.Presence == 1)]
print(f" -> layer 1: {len(known)} known hopper/band sightings, "
      f"{KNOWN_BUFFER_KM:.0f} km buffer")

# ---------------------------------------------------------------------
# GRID STATISTICS -- one cheap sweep to get the z-scoring reference
# ---------------------------------------------------------------------
srcs = {f: rasterio.open(os.path.join(RASTER_DIR, f + ".tif")) for f in ERA5_FEATURES}
ref = srcs[ERA5_FEATURES[0]]
W, H = ref.width, ref.height
print(f" -> grid {W} x {H} px, res {ref.res[0]:.4f} deg "
      f"(~{ref.res[0] * 111:.1f} km)   [ERA5 information is ~11 km]")

with rasterio.open(MASK_PATH) as msrc:
    mask_full = msrc.read(
        1, out_shape=(H, W), resampling=rasterio.enums.Resampling.nearest)

sums = {f: 0.0 for f in ERA5_FEATURES}
sqs = {f: 0.0 for f in ERA5_FEATURES}
cnt = 0
for i in range(0, H, BLOCK):
    for j in range(0, W, BLOCK):
        w, h = min(BLOCK, W - j), min(BLOCK, H - i)
        win = Window(j, i, w, h)
        arrs = {f: srcs[f].read(1, window=win).astype('float64') for f in ERA5_FEATURES}
        mk = mask_full[i:i + h, j:j + w]
        valid = (mk == 1)
        for f in ERA5_FEATURES:
            valid &= np.isfinite(arrs[f])
        n = int(valid.sum())
        if not n:
            continue
        cnt += n
        for f in ERA5_FEATURES:
            v = arrs[f][valid]
            sums[f] += v.sum()
            sqs[f] += (v ** 2).sum()

if cnt == 0:
    print("[FATAL] No valid pixels inside the habitat mask. Check the rasters align with the mask.")
    raise SystemExit(1)

grid_mu = {f: sums[f] / cnt for f in ERA5_FEATURES}
grid_sd = {f: max(np.sqrt(max(sqs[f] / cnt - (sums[f] / cnt) ** 2, 0.0)), 1e-9)
           for f in ERA5_FEATURES}
print(f" -> {cnt:,} valid habitat pixels; z-scoring against the grid's own distribution")
print(f"    {'feature':<16}{'grid mean':>14}{'grid SD':>12}{'train mean':>14}{'train SD':>12}")
for f in ERA5_FEATURES:
    print(f"    {f:<16}{grid_mu[f]:>14.4f}{grid_sd[f]:>12.4f}{tr_mu[f]:>14.4f}{tr_sd[f]:>12.4f}")
print("    ^ the grid/train gap is the approximation flagged in the docstring.")


def score_block(win, i, j, h, w):
    """Return (probs, valid_mask) for one block, z-scored against the grid."""
    arrs = {f: srcs[f].read(1, window=win).astype('float64') for f in ERA5_FEATURES}
    mk = mask_full[i:i + h, j:j + w]
    valid = (mk == 1)
    for f in ERA5_FEATURES:
        valid &= np.isfinite(arrs[f])
    if not valid.any():
        return None, valid
    # DataFrame, not ndarray -- keeps the column names the model was fitted with,
    # which both silences sklearn's warning and guarantees the feature order
    # cannot silently drift away from ERA5_FEATURES.
    X = pd.DataFrame({f: (arrs[f][valid] - grid_mu[f]) / grid_sd[f] for f in ERA5_FEATURES})
    return model.predict_proba(X)[:, 1], valid


# ---------------------------------------------------------------------
# PASS 1 -- learn the percentile thresholds
# ---------------------------------------------------------------------
print("\n[PASS 1] Sweeping the ROI to map the score distribution...")
rng = np.random.RandomState(42)
sample = []
for i in range(0, H, BLOCK):
    for j in range(0, W, BLOCK):
        w, h = min(BLOCK, W - j), min(BLOCK, H - i)
        probs, valid = score_block(Window(j, i, w, h), i, j, h, w)
        if probs is None:
            continue
        k = max(1, int(len(probs) * 0.05))
        sample.extend(rng.choice(probs, size=k, replace=False))

sample = np.asarray(sample)
thr = {k: float(np.percentile(sample, v)) for k, v in PCTL.items()}
print(f"  sampled {len(sample):,} pixel scores")
print(f"  HIGH   (top  5%): score >= {thr['high']:.4f}")
print(f"  MEDIUM (top 10%): score >= {thr['medium']:.4f}")
print(f"  LOW    (top 25%): score >= {thr['low']:.4f}")

# ---------------------------------------------------------------------
# PASS 2 -- classify, and build both layers
# ---------------------------------------------------------------------
print("\n[PASS 2] Classifying and writing layers...")
meta = ref.meta.copy()
meta.update(dtype=rasterio.uint8, nodata=255, count=1, compress='lzw')

lon0, lat0 = ref.transform.c, ref.transform.f
dlon, dlat = ref.transform.a, ref.transform.e

paths = {
    'model': os.path.join(RASTER_DIR, "risk_model_classes.tif"),
    'buffer': os.path.join(RASTER_DIR, "risk_known_buffer.tif"),
    'oper': os.path.join(RASTER_DIR, "risk_operational.tif"),
}
counts = {c: 0 for c in (0, 1, 2, 3)}
buf_px = 0
oper_counts = {c: 0 for c in (0, 1, 2, 3, 4)}

with rasterio.open(paths['model'], 'w', **meta) as dst_m, \
     rasterio.open(paths['buffer'], 'w', **meta) as dst_b, \
     rasterio.open(paths['oper'], 'w', **meta) as dst_o:

    for i in range(0, H, BLOCK):
        for j in range(0, W, BLOCK):
            w, h = min(BLOCK, W - j), min(BLOCK, H - i)
            win = Window(j, i, w, h)
            probs, valid = score_block(win, i, j, h, w)

            cls = np.full((h, w), 255, dtype=np.uint8)
            buf = np.full((h, w), 255, dtype=np.uint8)
            oper = np.full((h, w), 255, dtype=np.uint8)

            if probs is not None:
                c = np.zeros(len(probs), dtype=np.uint8)
                c[probs >= thr['low']] = 1
                c[probs >= thr['medium']] = 2
                c[probs >= thr['high']] = 3
                cls[valid] = c
                for k in (0, 1, 2, 3):
                    counts[k] += int((c == k).sum())

                # layer 1: distance from each valid pixel to nearest known sighting
                rr, cc = np.nonzero(valid)
                plat = lat0 + (i + rr + 0.5) * dlat
                plon = lon0 + (j + cc + 0.5) * dlon
                dmin = np.full(len(rr), np.inf)
                for _, s in known.iterrows():
                    d = haversine_km(plat, plon, s.Y, s.X)
                    np.minimum(dmin, d, out=dmin)
                inbuf = (dmin <= KNOWN_BUFFER_KM).astype(np.uint8)
                buf[valid] = inbuf
                buf_px += int(inbuf.sum())

                # combined operational map: buffer wins, model classes beyond it
                o = np.where(inbuf == 1, 4, c).astype(np.uint8)
                oper[valid] = o
                for k in (0, 1, 2, 3, 4):
                    oper_counts[k] += int((o == k).sum())

            dst_m.write(cls, 1, window=win)
            dst_b.write(buf, 1, window=win)
            dst_o.write(oper, 1, window=win)

for s in srcs.values():
    s.close()

# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------
px_km2 = (ref.res[0] * 111.0) * (ref.res[0] * 111.0)
print("\n" + "=" * 86)
print(" MAP COMPOSITION")
print("=" * 86)
print(f"  {'class':<28}{'pixels':>12}{'% of habitat':>14}{'area km2':>12}")
print(f"  {'-'*28}{'-'*12}{'-'*14}{'-'*12}")
names = {4: 'KNOWN buffer (<=30km)', 3: 'HIGH   (model, beyond)',
         2: 'MEDIUM (model, beyond)', 1: 'LOW    (model, beyond)',
         0: 'background'}
for k in (4, 3, 2, 1, 0):
    n = oper_counts[k]
    print(f"  {names[k]:<28}{n:>12,}{n / cnt:>13.1%}{n * px_km2:>12,.0f}")

actionable = oper_counts[3] + oper_counts[2]
print(f"\n  Model-flagged HIGH+MEDIUM beyond the known buffer: "
      f"{actionable:,} px ({actionable / cnt:.1%} of habitat, {actionable * px_km2:,.0f} km2)")
print("  That is the surveyable area the model adds on top of what FAO already knows.")

summary = {
    'test_anchor': str(TEST_ANCHOR_DATE.date()),
    'model': 'ERA5-only (5 features), z-scored within anchor, RF untuned, threshold-free (percentile)',
    'train_weeks': REANCHOR_TRAIN_WEEKS,
    'n_train_points': int(len(train_pool)),
    'n_known_sightings': int(len(known)),
    'known_buffer_km': KNOWN_BUFFER_KM,
    'grid': {'width': W, 'height': H, 'res_deg': float(ref.res[0]),
             'valid_habitat_px': int(cnt), 'px_km2': float(px_km2)},
    'grid_zscore_reference': {f: {'mean': grid_mu[f], 'sd': grid_sd[f]} for f in ERA5_FEATURES},
    'train_zscore_reference': {f: {'mean': float(tr_mu[f]), 'sd': float(tr_sd[f])}
                               for f in ERA5_FEATURES},
    'percentile_thresholds': thr,
    'model_class_pixels': counts,
    'operational_class_pixels': oper_counts,
    # basenames only -- an absolute path here would leak the machine it was
    # generated on into a committed file and make git diff noisy for everyone.
    'outputs': {k: os.path.basename(v) for k, v in paths.items()},
}
json_path = os.path.join(RASTER_DIR, "risk_map_thresholds.json")
with open(json_path, 'w') as fh:
    json.dump(summary, fh, indent=2)

print("\n" + "=" * 86)
print("  Written:")
for k, v in paths.items():
    print(f"    {os.path.basename(v)}")
print(f"    {os.path.basename(json_path)}")
print("\n  PRESENT THE TWO-LAYER VERSION. risk_operational.tif marks the")
print("  known-activity buffer as class 4 so it is visually separable -- presenting a")
print("  single blended surface would take credit for proximity the model did not earn.")
print("=" * 86)
print("\nNext: run 07_validate_risk_map.py to score it against the real April swarms.")
