"""
Scores the ORIGINAL trained model (fit on Kenya/Ethiopia train_pool/, fresh
here with the same spec: RandomForest, 100 trees, depth 5, balanced classes,
random_state=42, threshold 0.5, nothing tuned) against the Somaliland point
set, for both ERA5-only and ERA5+Sentinel-1 configurations, with and without
within-anchor seasonal (z-score) correction. Mirrors 04_train_and_test.py's
scoring block exactly -- same zscore() function, same metric definitions,
same distance baseline and paired-bootstrap CI logic. The metrics are computed
on the same scale, while the smaller sample and reconstructed habitat mask
limit substantive comparison with the headline result.

IMPORTANT READING NOTE
Somaliland absences here are entirely synthetic (0 real FAO 'NO LOCUST'
records fall in this box in this window -- see S1's output). The headline
W7-W10 swarm test is also synthetic-only; the main W4-W5 diagnostic is the
separate analysis that compares against real surveyed absences.

OUTPUT: results_somaliland.csv, printed summary table
"""

import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

from somaliland_config import (DATA_DIR, MAIN_DATA_DIR, TRAIN_WEEKS,
                               ALL_FEATURES, ERA5_FEATURES, CONFIGS,
                               THRESHOLD, haversine_km, week_bounds)

print("[START] S3 - Testing the trained model on Somaliland hopper+band sightings")

TRAIN_PATH = os.path.join(MAIN_DATA_DIR, "train_pool", "features_all.csv")
SOM_PATH = os.path.join(DATA_DIR, "features_all.csv")

for p, hint in ((TRAIN_PATH, "This should already exist from the main pipeline "
                             "(01, 02a-c --set train, 03 --set train)."),
                (SOM_PATH, "Run S1_prepare_points.py then S2_extract_features.py first.")):
    if not os.path.exists(p):
        print(f"[FATAL] {p} not found.\n        {hint}")
        raise SystemExit(1)

train_all = pd.read_csv(TRAIN_PATH)
som_all = pd.read_csv(SOM_PATH)

train_pool = train_all[train_all.Week.isin(TRAIN_WEEKS)]
print(f" -> train (Kenya/Ethiopia): {len(train_pool)} points, "
      f"{int((train_pool.Presence == 1).sum())} hopper/band sightings")
print(f" -> test  (Somaliland)    : {len(som_all)} points, "
      f"{int((som_all.Presence == 1).sum())} hopper/band sightings "
      f"({int((som_all.Source=='synthetic_background').sum())} synthetic absences, "
      f"{int((som_all.Source=='fao_no_locust').sum())} real absences)\n")


def zscore(df, ref, cols):
    out = df.copy()
    for c in cols:
        mu, sd = ref[c].mean(), ref[c].std()
        out[c] = (df[c] - mu) / (sd if sd else 1.0)
    return out


def make_model():
    return RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                  max_depth=5, random_state=42)


rows = []
for corrected in (True, False):
    tr_src = zscore(train_all, train_all, ALL_FEATURES) if corrected else train_all
    te_src = zscore(som_all, som_all, ALL_FEATURES) if corrected else som_all
    tag_c = "with seasonal correction" if corrected else "without correction"

    for label, cols, tag in CONFIGS:
        tr = tr_src[tr_src.Week.isin(TRAIN_WEEKS)].dropna(subset=cols)
        model = make_model()
        model.fit(tr[cols], tr.Presence)

        d = te_src.dropna(subset=cols)
        if len(d) == 0 or d.Presence.nunique() < 2:
            print(f" [SKIP] {label} / {tag_c}: not enough class variety to score "
                  f"(n={len(d)}, positives={int(d.Presence.sum()) if len(d) else 0})")
            continue

        proba = model.predict_proba(d[cols])[:, 1]
        y = d.Presence.values
        tn, fp, fn, tp = confusion_matrix(
            y, (proba >= THRESHOLD).astype(int), labels=[0, 1]).ravel()
        auc = roc_auc_score(y, proba)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0

        print(f" {label:<32}{tag_c:<28}AUC {auc:>6.3f}   "
              f"PRECISION {prec:>6.1%}   RECALL {rec:>6.1%}   "
              f"found {tp}/{tp+fn}   false_alarms {fp}   (n={len(d)})")

        rows.append({'seasonal_correction': corrected, 'features': tag,
                     'n': len(d), 'positives': int(y.sum()), 'auc': round(auc, 4),
                     'precision': round(prec, 4), 'recall': round(rec, 4),
                     'found': int(tp), 'missed': int(fn), 'false_alarms': int(fp)})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(DATA_DIR, "results_somaliland.csv"), index=False)

# ---------------------------------------------------------------------
# DISTANCE BASELINE -- distance to nearest ORIGINAL (Kenya/Ethiopia)
# training hopper/band site. If Somaliland points score no better than
# this, the model is not doing anything a lookup table couldn't.
# ---------------------------------------------------------------------
print("\n" + "=" * 88)
print(" DISTANCE BASELINE -- distance to nearest Kenya/Ethiopia training site")
print("=" * 88)

known = train_all[train_all.Week.isin(TRAIN_WEEKS) & (train_all.Presence == 1)]
tez = zscore(som_all, som_all, ERA5_FEATURES)
trz = zscore(train_all, train_all, ERA5_FEATURES)
trh = trz[trz.Week.isin(TRAIN_WEEKS)].dropna(subset=ERA5_FEATURES)
m_head = make_model()
m_head.fit(trh[ERA5_FEATURES], trh.Presence)

d = tez.dropna(subset=ERA5_FEATURES)
if len(d) and d.Presence.nunique() >= 2:
    dist = np.array([haversine_km(r.Y, r.X, known.Y.values, known.X.values).min()
                     for _, r in d.iterrows()])
    proba = m_head.predict_proba(d[ERA5_FEATURES])[:, 1]
    y = d.Presence.values

    a_b = roc_auc_score(y, -dist)
    a_m = roc_auc_score(y, proba)

    N_BOOT = 2000
    BOOT_RNG = np.random.RandomState(42)

    def paired_bootstrap_auc_diff(y, s_a, s_b, n=N_BOOT):
        idx = np.arange(len(y))
        diffs = []
        for _ in range(n):
            b = BOOT_RNG.choice(idx, size=len(idx), replace=True)
            if len(np.unique(y[b])) < 2:
                continue
            diffs.append(roc_auc_score(y[b], s_a[b]) - roc_auc_score(y[b], s_b[b]))
        dd = np.array(diffs)
        return float(dd.mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))

    diff, lo, hi = paired_bootstrap_auc_diff(y, proba, -dist)
    verdict = ("baseline wins" if hi < 0 else
               "model wins" if lo > 0 else "indistinguishable")

    print(f" model (ERA5-only) AUC {a_m:.3f}   distance-baseline AUC {a_b:.3f}   "
          f"difference {a_m - a_b:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   {verdict.upper()}")
    print(f"\n Distances range {dist.min():.0f}-{dist.max():.0f} km from the nearest "
          f"Kenya/Ethiopia training site (median {np.median(dist):.0f} km). If the "
          f"distance baseline AUC sits near 0.5, that is expected -- all Somaliland "
          f"points are roughly equally far from the training region, so distance alone "
          f"carries little separating information here, unlike the within-region swarm "
          f"test where distance is informative.")

    pd.DataFrame([{'comparison': 'model_vs_distance_baseline_somaliland', 'n': int(len(d)),
                   'auc_model': round(float(a_m), 4), 'auc_baseline': round(float(a_b), 4),
                   'auc_difference': round(float(a_m - a_b), 4),
                   'ci_low': round(lo, 4), 'ci_high': round(hi, 4),
                   'verdict': verdict, 'n_bootstrap': N_BOOT}]).to_csv(
        os.path.join(DATA_DIR, "baseline_comparison_somaliland.csv"), index=False)
else:
    print(" [SKIP] not enough class variety to run the distance baseline.")

print(f"\n[SUCCESS] Saved results_somaliland.csv"
      + (" and baseline_comparison_somaliland.csv" if len(d) and d.Presence.nunique() >= 2 else ""))
