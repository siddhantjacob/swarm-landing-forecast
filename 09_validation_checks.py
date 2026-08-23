"""
Six validation diagnostics. No GEE, runs in seconds.

    1  optimism gap      random k-fold against forward temporal holdout
    2  spatial block CV  whole 0.5 degree blocks held out
    3  Moran's I         on model RESIDUALS, not on the prediction surface
    4  effective n       distinct ERA5 cells occupied, against sighting count
    5  absence type      real surveyed against synthetic, on W4-W5
    6  training period   the optimism gap expressed in recall

Random cross-validation places spatially and temporally adjacent
observations in both folds, so with an autocorrelated response it measures
interpolation rather than prediction (Roberts et al. 2017, Ecography
40:913-929; Ploton et al. 2020, Nature Communications 11:4540).

OUTPUT: validation_checks.csv
"""

import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

from timeline_config import (RESULTS_DIR, BASE_DIR, DATA_DIR, ERA5_FEATURES, ALL_FEATURES, THRESHOLD,
                             haversine_km)

TRAIN_WEEKS = [1, 2, 3, 4, 5]
TEST_WEEKS = [7, 8, 9, 10]
ERA5_CELL_DEG = 0.1        # ERA5-Land native grid
BLOCK_DEG = 0.5            # ~55 km spatial blocks for block-CV
MORAN_BAND_KM = 30.0       # neighbours within this distance
N_PERM = 999
RNG = np.random.RandomState(42)

TRAIN_PATH = os.path.join(DATA_DIR, "train_pool", "features_all.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_anchor", "features_all.csv")

print("[START] 09 - Validation diagnostics\n")
for p in (TRAIN_PATH, TEST_PATH):
    if not os.path.exists(p):
        print(f"[FATAL] {p} not found. Run 01 -> 02 -> 03 first.")
        raise SystemExit(1)

train_all = pd.read_csv(TRAIN_PATH)
test_all = pd.read_csv(TEST_PATH)
test_all = test_all[(test_all.Presence == 0) | (test_all.Source == 'fao_swarm')]


def zscore(df, ref, cols):
    out = df.copy()
    for c in cols:
        mu, sd = ref[c].mean(), ref[c].std()
        out[c] = (df[c] - mu) / (sd if sd else 1.0)
    return out


def make_model():
    return RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                  max_depth=5, random_state=42)


trz = zscore(train_all, train_all, ALL_FEATURES)
tez = zscore(test_all, test_all, ALL_FEATURES)
tr = trz[trz.Week.isin(TRAIN_WEEKS)].dropna(subset=ERA5_FEATURES).reset_index(drop=True)
te = tez[tez.Week.isin(TEST_WEEKS)].dropna(subset=ERA5_FEATURES).reset_index(drop=True)

X, y = tr[ERA5_FEATURES].values, tr.Presence.values
print(f" -> training set: {len(tr)} points, {int(y.sum())} sightings")
print(f" -> holdout set : {len(te)} points, {int(te.Presence.sum())} sightings\n")

rows = []

# =====================================================================
# 4. EFFECTIVE SAMPLE SIZE  (run first -- it frames everything else)
# =====================================================================
print("=" * 82)
print(" 4. EFFECTIVE SAMPLE SIZE -- how much independent information is there?")
print("=" * 82)
for name, d in (("training", tr), ("holdout", te)):
    pos = d[d.Presence == 1]
    cells = set(zip((pos.X / ERA5_CELL_DEG).round().astype(int),
                    (pos.Y / ERA5_CELL_DEG).round().astype(int)))
    ratio = len(pos) / max(1, len(cells))
    print(f"  {name:<10}{len(pos):>5} sightings occupy {len(cells):>4} distinct ERA5 cells "
          f"({ERA5_CELL_DEG}deg, ~11 km)  ->  {ratio:.1f} sightings per cell")
    rows.append({'check': 'effective_n', 'set': name, 'metric': 'sightings',
                 'value': len(pos)})
    rows.append({'check': 'effective_n', 'set': name, 'metric': 'distinct_era5_cells',
                 'value': len(cells)})
print("\n  Sightings sharing a cell share an identical feature vector, so the")
print("  cell count is the effective denominator.")

# =====================================================================
# 1. THE OPTIMISM GAP
# =====================================================================
print("\n" + "=" * 82)
print(" 1. OPTIMISM GAP -- random k-fold vs forward temporal holdout")
print("=" * 82)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf_auc = []
oof_kf = np.full(len(y), np.nan)          # out-of-fold scores, kept for check 5
for tr_i, te_i in skf.split(X, y):
    m = make_model(); m.fit(X[tr_i], y[tr_i])
    oof_kf[te_i] = m.predict_proba(X[te_i])[:, 1]
    kf_auc.append(roc_auc_score(y[te_i], oof_kf[te_i]))

m_full = make_model(); m_full.fit(X, y)
# .values on both sides: m_full was fitted on a bare array, so passing a
# DataFrame here triggers a sklearn feature-names warning on every call.
fwd = roc_auc_score(te.Presence, m_full.predict_proba(te[ERA5_FEATURES].values)[:, 1])

print(f"  Random 5-fold CV (literature-standard) : {np.mean(kf_auc):.3f} "
      f"(folds {', '.join(f'{a:.3f}' for a in kf_auc)})")
print(f"  Forward temporal holdout               : {fwd:.3f}")
print(f"  OPTIMISM GAP                           : {np.mean(kf_auc) - fwd:+.3f}")
rows += [{'check': 'optimism', 'set': 'random_kfold', 'metric': 'auc',
          'value': round(float(np.mean(kf_auc)), 4)},
         {'check': 'optimism', 'set': 'forward_holdout', 'metric': 'auc',
          'value': round(float(fwd), 4)}]

# =====================================================================
# 2. SPATIAL BLOCK CV
# =====================================================================
print("\n" + "=" * 82)
print(f" 2. SPATIAL BLOCK CV -- {BLOCK_DEG} deg blocks (~{BLOCK_DEG*111:.0f} km), whole blocks held out")
print("=" * 82)

blocks = (tr.X / BLOCK_DEG).round().astype(int).astype(str) + "_" + \
         (tr.Y / BLOCK_DEG).round().astype(int).astype(str)
n_blocks = blocks.nunique()
n_folds = min(5, n_blocks)
gkf = GroupKFold(n_splits=n_folds)
bl_auc = []
oof_bl = np.full(len(y), np.nan)          # out-of-fold scores, kept for check 5
for tr_i, te_i in gkf.split(X, y, groups=blocks):
    m = make_model(); m.fit(X[tr_i], y[tr_i])
    oof_bl[te_i] = m.predict_proba(X[te_i])[:, 1]
    if len(np.unique(y[te_i])) < 2:
        continue
    bl_auc.append(roc_auc_score(y[te_i], oof_bl[te_i]))

print(f"  {n_blocks} spatial blocks, {n_folds}-fold grouped CV")
if bl_auc:
    print(f"  Spatial block CV AUC : {np.mean(bl_auc):.3f} "
          f"(folds {', '.join(f'{a:.3f}' for a in bl_auc)})")
    rows.append({'check': 'spatial_block_cv', 'set': 'blocks', 'metric': 'auc',
                 'value': round(float(np.mean(bl_auc)), 4)})
    # Ranked by measured value, not assumed order -- block CV can fall
    # below the forward holdout.
    ranked = sorted([('random k-fold', float(np.mean(kf_auc))),
                     ('spatial block CV', float(np.mean(bl_auc))),
                     ('forward holdout', float(fwd))],
                    key=lambda t: -t[1])
    print("\n  " + "  >  ".join(f"{n} {v:.3f}" for n, v in ranked))
    if np.mean(bl_auc) < np.mean(kf_auc) - 0.03:
        print("  Block CV falls below random k-fold, confirming that random CV was")
        print("  inflated by spatial adjacency between training and test folds.")
    else:
        print("  Block CV is close to random k-fold, so spatial adjacency alone does")
        print("  not explain the gap -- the temporal split is doing the work.")
    if np.mean(bl_auc) < fwd:
        print("  Block CV is the most pessimistic scheme here, below the forward")
        print("  holdout: holding out whole regions is harder than holding out later")
        print("  weeks.")
else:
    print("  [SKIP] blocks too imbalanced for grouped CV")

# =====================================================================
# 3. MORAN'S I ON RESIDUALS
# =====================================================================
print("\n" + "=" * 82)
print(f" 3. MORAN'S I ON RESIDUALS -- binary weights within {MORAN_BAND_KM:.0f} km, "
      f"{N_PERM} permutations")
print("=" * 82)


def morans_i(lat, lon, z, band_km=MORAN_BAND_KM, n_perm=N_PERM):
    """Global Moran's I with binary distance-band weights; permutation p."""
    n = len(z)
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        d = haversine_km(lat[i], lon[i], lat, lon)
        W[i] = (d <= band_km)
    np.fill_diagonal(W, 0.0)
    s0 = W.sum()
    if s0 == 0:
        return np.nan, np.nan, 0
    dev = z - z.mean()
    denom = (dev ** 2).sum()
    if denom == 0:
        return np.nan, np.nan, int(s0)

    def I_of(v):
        dv = v - v.mean()
        return (n / s0) * (dv @ W @ dv) / (dv ** 2).sum()

    obs = I_of(z)
    perm = np.array([I_of(RNG.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    return float(obs), float(p), int(s0)


print(f"  {'set':<22}{'n':>6}{'Moran I':>10}{'p':>9}   interpretation")
print("  " + "-" * 74)
for name, d, feats in (("training residuals", tr, ERA5_FEATURES),
                       ("holdout residuals", te, ERA5_FEATURES)):
    m = make_model()
    if name.startswith("training"):
        m.fit(X, y)
        resid = d.Presence.values - m.predict_proba(d[feats].values)[:, 1]
    else:
        resid = d.Presence.values - m_full.predict_proba(d[feats].values)[:, 1]
    I, p, _ = morans_i(d.Y.values, d.X.values, resid)
    if not np.isfinite(I):
        continue
    verdict = ("clustered residuals -- unmodelled spatial structure"
               if (p < 0.05 and I > 0) else
               "no significant residual clustering")
    print(f"  {name:<22}{len(d):>6}{I:>10.4f}{p:>9.3f}   {verdict}")
    rows.append({'check': 'morans_i_residuals', 'set': name, 'metric': 'I',
                 'value': round(I, 4)})
    rows.append({'check': 'morans_i_residuals', 'set': name, 'metric': 'p',
                 'value': round(p, 4)})

print("\n  Positive and significant means the model leaves spatial structure behind,")
print("  the independence assumption is violated, and reported AUC is optimistic.")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS_DIR, "validation_checks.csv"), index=False)
print(f"\n[SUCCESS] Saved validation_checks.csv")


# =====================================================================
# 5. REAL vs SYNTHETIC ABSENCES -- the sharpest criticism of this design
# =====================================================================
# Every metric on the W7-W10 test weeks is computed against SYNTHETIC
# background: those weeks contain 480 absences, all of them habitat-masked
# random points and none of them surveyed. The 51 real FAO 'NO LOCUST'
# records fall in W4 and W5.
#
# That matters most for the DISTANCE BASELINE. Uniformly-sampled points are
# by construction far from clustered sightings, so "distance to nearest known
# sighting" separates them almost trivially. A real surveyed absence is a
# place someone went and found nothing -- usually because they were looking
# near an infestation, so it sits CLOSE to known activity. If the baseline's
# advantage is an artefact of uniform sampling, it should shrink or vanish
# when scored against real absences.
#
# W4-W5 of the TRAINING point set is the only place both absence types exist
# together, so the test runs there: train on W1-W3, test on W4-W5, scoring
# the model and the baseline separately against real and synthetic absences.
# The train anchor's T0 window closes 2020-02-18, well before W4 opens on
# 2020-03-10, so there is no leakage.
#
# n is small: 79 sightings, 51 real absences.
# =====================================================================
print("\n" + "=" * 82)
print(" 5. REAL vs SYNTHETIC ABSENCES -- is the distance baseline an artefact?")
print("=" * 82)

sub = trz[trz.Week.isin([4, 5])].dropna(subset=ERA5_FEATURES)
fit3 = trz[trz.Week.isin([1, 2, 3])].dropna(subset=ERA5_FEATURES)
known3 = train_all[train_all.Week.isin([1, 2, 3]) & (train_all.Presence == 1)]

if len(fit3) >= 30 and fit3.Presence.nunique() == 2 and len(known3):
    m3 = make_model(); m3.fit(fit3[ERA5_FEATURES], fit3.Presence)
    pos = sub[sub.Presence == 1]
    print(f"  train W1-W3: {int(fit3.Presence.sum())} sightings   "
          f"test W4-W5: {len(pos)} sightings")
    print(f"\n  {'absences used':<24}{'n abs':>7}{'model AUC':>12}{'baseline AUC':>15}"
          f"{'baseline lead':>15}")
    print("  " + "-" * 73)

    for lbl, src in (("REAL surveyed only", 'fao_no_locust'),
                     ("synthetic only", 'synthetic_background')):
        neg = sub[sub.Source == src]
        if len(neg) < 10:
            print(f"  {lbl:<24}{len(neg):>7}   [too few to score]")
            continue
        d = pd.concat([pos, neg])
        y = d.Presence.values
        p_model = m3.predict_proba(d[ERA5_FEATURES])[:, 1]
        dist = np.array([haversine_km(r.Y, r.X, known3.Y.values, known3.X.values).min()
                         for _, r in d.iterrows()])
        a_m = roc_auc_score(y, p_model)
        a_b = roc_auc_score(y, -dist)
        print(f"  {lbl:<24}{len(neg):>7}{a_m:>12.3f}{a_b:>15.3f}{a_b - a_m:>+15.3f}")
        rows.append({'check': 'absence_type', 'set': lbl, 'metric': 'model_auc',
                     'value': round(float(a_m), 4)})
        rows.append({'check': 'absence_type', 'set': lbl, 'metric': 'baseline_auc',
                     'value': round(float(a_b), 4)})

    print("\n  If the baseline's lead SHRINKS on real absences, its advantage in the")
    print("  main results is partly an artefact of uniform pseudo-absence sampling and")
    print("  must be qualified. If the lead HOLDS, the finding is robust to absence")
    print("  definition and the criticism is answered.")
else:
    print("  [SKIP] insufficient W1-W3 data")

# =====================================================================
# 6. TRAINING-PERIOD PERFORMANCE -- the optimism gap in RECALL units
# =====================================================================
# Checks 1 and 2 give the optimism gap in AUC; the same gap in recall, at
# the fixed 0.5 threshold. Four schemes, one model, one feature set:
#
#   in-sample      fitted and scored on the SAME points. This is what the
#                  model memorised. It is NOT a performance figure and is
#                  printed only so the distance to the others is visible.
#   random k-fold  the literature-standard protocol
#   forward        train on W1-W5 bands, predict W7-W10 swarms -- the design
#   block CV       whole 0.5 deg regions held out
#
# WHAT TO LOOK FOR. If block-CV recall falls BELOW the forward-holdout
# recall, then removing spatial proximity costs more than forecasting six
# weeks into the future -- i.e. the model is substantially a proximity
# model, and the forward holdout is flattered by the test swarms sitting in
# the same places as the training bands. That is the same conclusion the
# distance baseline (04) and Moran's I (check 3) reach by other routes.
#
# NOTE ON POOLING. Recall here is POOLED over all points, whereas 04 reports
# the mean of four per-week recalls. Both are correct; they differ slightly
# because the weeks carry different numbers of sightings. The AUCs printed
# here are likewise pooled out-of-fold and will sit a little below the
# mean-of-folds AUCs in checks 1 and 2 -- folds are calibrated differently,
# so pooling their scores into one ranking loses a little separation. Quote
# checks 1 and 2 for AUC and this table for recall.
print("\n" + "=" * 82)
print(" 6. TRAINING-PERIOD PERFORMANCE -- the optimism gap in RECALL")
print("=" * 82)


def prf(y_true, score):
    pred = (score >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, score) if len(np.unique(y_true)) > 1 else np.nan
    return (auc, tp / (tp + fn) if (tp + fn) else 0.0,
            tp / (tp + fp) if (tp + fp) else 0.0, int(tp), int(tp + fn))


y_tr = tr.Presence.values
in_s = m_full.predict_proba(X)[:, 1]
fwd_s = m_full.predict_proba(te[ERA5_FEATURES].values)[:, 1]

schemes = [("in-sample (memorised, NOT a result)", y_tr, in_s),
           ("random 5-fold CV", y_tr, oof_kf),
           ("FORWARD HOLDOUT (W7-W10 swarms)", te.Presence.values, fwd_s),
           ("spatial block CV", y_tr, oof_bl)]

print(f"  {'scheme':<38}{'AUC':>7}{'RECALL':>9}{'PRECIS':>9}{'found':>12}")
print("  " + "-" * 76)
for name, yy, ss in schemes:
    ok = np.isfinite(ss)
    if ok.sum() == 0 or len(np.unique(yy[ok])) < 2:
        print(f"  {name:<38}{'[skipped]':>37}")
        continue
    a, r, p, tp, n = prf(yy[ok], ss[ok])
    print(f"  {name:<38}{a:>7.3f}{r:>9.1%}{p:>9.1%}{f'{tp}/{n}':>12}")
    rows.append({'check': 'training_period', 'set': name, 'metric': 'recall',
                 'value': round(float(r), 4)})
    rows.append({'check': 'training_period', 'set': name, 'metric': 'auc_pooled',
                 'value': round(float(a), 4)})

# The comparison the section exists for.
_, r_fwd, _, _, _ = prf(te.Presence.values, fwd_s)
ok = np.isfinite(oof_bl)
if ok.sum() and len(np.unique(y_tr[ok])) > 1:
    _, r_blk, _, _, _ = prf(y_tr[ok], oof_bl[ok])
    print(f"\n  block CV recall {r_blk:.1%}  vs  forward holdout recall {r_fwd:.1%}")
    if r_blk < r_fwd:
        print("  Removing spatial proximity costs MORE than forecasting six weeks")
        print("  forward. The model is substantially a proximity model, and the")
        print("  forward holdout is flattered by the April swarms occupying much the")
        print("  same ground as the February-March bands. Same conclusion as the")
        print("  distance baseline in 04 and Moran's I in check 3 above.")
    else:
        print("  Block CV holds up against the forward holdout, so spatial proximity")
        print("  is not carrying the forward result.")

# Per-week in-sample recall, showing how unevenly the training signal is
# distributed across the five weeks.
print(f"\n  per training week, in-sample recall:")
print(f"    {'week':<7}{'sightings':>11}{'% of train':>12}{'recall':>9}")
print("    " + "-" * 39)
tr_pos_total = int(y_tr.sum())
for wk in TRAIN_WEEKS:
    d = tr[tr.Week == wk]
    pos = d[d.Presence == 1]
    if not len(pos):
        continue
    hit = int((m_full.predict_proba(pos[ERA5_FEATURES].values)[:, 1] >= THRESHOLD).sum())
    share = len(pos) / tr_pos_total
    print(f"    W{wk:<6}{len(pos):>11}{share:>12.1%}{hit / len(pos):>9.1%}")
    rows.append({'check': 'train_week_balance', 'set': f'W{wk}', 'metric': 'sightings',
                 'value': int(len(pos))})
top = max(TRAIN_WEEKS, key=lambda w: int((tr.Week == w).mul(tr.Presence).sum()))
top_share = int((tr.Week == top).mul(tr.Presence).sum()) / tr_pos_total
if top_share > 0.30:
    print(f"\n  W{top} alone holds {top_share:.0%} of the training sightings. The training")
    print("  training period is dominated by its heaviest week.")

pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "validation_checks.csv"), index=False)
print("\nNext: run 10_plot_wind.py")
