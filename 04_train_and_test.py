"""
Trains on hopper and band sightings from W1-W5 (features anchored
2020-02-03) and scores April swarms W7-W10 one week at a time (features
anchored 2020-03-15). Forecast lead 16 / 23 / 30 / 37 days.

Random Forest, 100 trees, depth 5, balanced classes, threshold 0.5. Nothing
is tuned anywhere.

Features are z-scored within their own anchor, so a value means "warmer or
cooler than normal for this time of year" rather than an absolute
temperature. The region warms ~2 K between anchors while sighting and
non-sighting locations differ by ~4 K, so without this the model reads every
March point as too warm. Results are reported both ways.

Also reported: SHAP, as fitted on the training points and as applied to the
test points; a distance baseline using no satellite data; a paired bootstrap
95% CI on the difference between model and baseline; and a logistic
regression on the same features.

    --with-smap        adds two SMAP features and two further configurations
    --target presence  scores all locust categories instead of swarms only

OUTPUT: results_weekly.csv, feature_importance.csv, shap_values.csv,
        fig_shap_summary.png, baseline_comparison.csv
"""

import pandas as pd
import numpy as np
import os
import inspect

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

from timeline_config import (RESULTS_DIR, BASE_DIR, DATA_DIR, ANCHOR_DATE, TEST_ANCHOR_DATE,
                             CONFIGS, CONFIGS_SMAP, ERA5_FEATURES, ALL_FEATURES,
                             ALL_PLUS_SMAP, THRESHOLD,
                             week_bounds, test_lead_days, haversine_km)

TRAIN_WEEKS = [1, 2, 3, 4, 5]
TEST_WEEKS = [7, 8, 9, 10]

import sys
# --target presence  scores ALL locust categories instead of swarms only.
# That is the target the published literature uses (Klein AUC 0.761,
# Dynamic Forecast AUC 0.767), so it is how this study compares to them.
# It is an EASIER target: the features represent environmental suitability
# associated with breeding habitat, and hoppers hatch out of that habitat,
# whereas swarms may fly in from elsewhere.
PRESENCE = '--target' in sys.argv and 'presence' in sys.argv
TARGET = 'presence (all categories)' if PRESENCE else 'swarms only'

# --with-smap adds two dedicated L-band soil moisture features and two extra
# configs, so the Sentinel-1 backscatter proxy can be compared against a real
# soil moisture retrieval. Requires 02d_extract_smap.py + a 03 re-run.
WITH_SMAP = '--with-smap' in sys.argv
CFG = CONFIGS_SMAP if WITH_SMAP else CONFIGS
FEATS = ALL_PLUS_SMAP if WITH_SMAP else ALL_FEATURES

TRAIN_PATH = os.path.join(DATA_DIR, "train_pool", "features_all.csv")
TEST_PATH = os.path.join(DATA_DIR,
                         "presence_anchor" if PRESENCE else "test_anchor",
                         "features_all.csv")

print(f"[START] 04 - Train on hopper bands, test on April {TARGET}")
print(f"          train features anchored {ANCHOR_DATE.date()}, "
      f"test features anchored {TEST_ANCHOR_DATE.date()}\n")

for p, hint in ((TRAIN_PATH, "Run 01, then 02a/02b/02c --set train, then 03 --set train."),
                (TEST_PATH, "Run 02a/02b/02c --set presence, then 03 --set presence."
                             if PRESENCE else
                             "Run 02a/02b/02c --anchor test, then 03 --anchor test.")):
    if not os.path.exists(p):
        print(f"[FATAL] {p} not found.\n        {hint}")
        raise SystemExit(1)

train_all = pd.read_csv(TRAIN_PATH)
test_all = pd.read_csv(TEST_PATH)

if WITH_SMAP:
    missing = [c for c in FEATS if c not in train_all.columns or c not in test_all.columns]
    if missing:
        print(f"[FATAL] --with-smap needs {missing}, which are not in the feature files.")
        print("        Run:  python 02d_extract_smap.py --set train")
        print("              python 02d_extract_smap.py --anchor test")
        print("        then re-run 03_merge_features.py in BOTH modes.")
        raise SystemExit(1)
    print(" -> SMAP L-band soil moisture included (4 configs will be compared)\n")

if not PRESENCE:
    # Swarms only. Scattered adults near a training site may be that same
    # cohort after fledging, so they are excluded as a precaution.
    test_all = test_all[(test_all.Presence == 0) | (test_all.Source == 'fao_swarm')].copy()

train_pool = train_all[train_all.Week.isin(TRAIN_WEEKS)]
print(f" -> train: {len(train_pool)} points, "
      f"{int((train_pool.Presence == 1).sum())} hopper/band sightings")
print(f" -> test : {int((test_all[test_all.Week.isin(TEST_WEEKS)].Presence == 1).sum())} "
      f"{TARGET} sightings across W7-W10\n")


def zscore(df, ref, cols):
    """Standardise `cols` against the distribution of `ref` (same anchor)."""
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
    tr_src = zscore(train_all, train_all, FEATS) if corrected else train_all
    te_src = zscore(test_all, test_all, FEATS) if corrected else test_all
    tag_c = "with seasonal correction" if corrected else "without correction"

    for label, cols, tag in CFG:
        tr = tr_src[tr_src.Week.isin(TRAIN_WEEKS)].dropna(subset=cols)
        model = make_model()
        model.fit(tr[cols], tr.Presence)

        print("=" * 88)
        print(f" {label}  --  {tag_c}")
        print("=" * 88)
        print(f" {'week':<6}{'dates':<26}{'lead':>6}{'sights':>8}{'AUC':>8}"
              f"{'PRECISION':>11}{'RECALL':>9}{'found':>10}{'missed':>8}")
        print(" " + "-" * 86)

        for wk in TEST_WEEKS:
            d = te_src[te_src.Week == wk].dropna(subset=cols)
            if len(d) == 0 or d.Presence.nunique() < 2:
                continue
            proba = model.predict_proba(d[cols])[:, 1]
            y = d.Presence.values
            tn, fp, fn, tp = confusion_matrix(
                y, (proba >= THRESHOLD).astype(int), labels=[0, 1]).ravel()
            auc = roc_auc_score(y, proba)
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            s, e = week_bounds(wk)

            print(f" W{wk:<5}{str(s.date()) + '..' + str(e.date()):<26}"
                  f"{str(test_lead_days(wk)) + 'd':>6}{int(y.sum()):>8}{auc:>8.3f}"
                  f"{prec:>11.1%}{rec:>9.1%}"
                  f"{str(tp) + '/' + str(tp + fn):>10}{fn:>8}")

            rows.append({'seasonal_correction': corrected, 'features': tag,
                         'week': wk, 'start': str(s.date()), 'end': str(e.date()),
                         'lead_days': test_lead_days(wk), 'swarms': int(y.sum()),
                         'auc': round(auc, 4), 'precision': round(prec, 4),
                         'recall': round(rec, 4), 'found': int(tp), 'missed': int(fn),
                         'false_alarms': int(fp)})
        print()

out = pd.DataFrame(rows)
suffix = ("_presence" if PRESENCE else "") + ("_smap" if WITH_SMAP else "")
out.to_csv(os.path.join(RESULTS_DIR, f"results_weekly{suffix}.csv"), index=False)

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------
print("=" * 88)
print(" SUMMARY (mean over W7-W10)")
print("=" * 88)
print(f" {'features':<34}{'correction':<20}{'AUC':>8}{'PRECISION':>12}{'RECALL':>9}")
print(" " + "-" * 86)
for corrected in (True, False):
    for label, cols, tag in CFG:
        g = out[(out.seasonal_correction == corrected) & (out.features == tag)]
        if not len(g):
            continue
        print(f" {label:<34}{('yes' if corrected else 'no'):<20}"
              f"{g.auc.mean():>8.3f}{g.precision.mean():>12.1%}{g.recall.mean():>9.1%}")

# ---------------------------------------------------------------------
# FEATURE IMPORTANCE (best config)
# ---------------------------------------------------------------------
trz = zscore(train_all, train_all, FEATS)
tr = trz[trz.Week.isin(TRAIN_WEEKS)].dropna(subset=FEATS)
m = make_model(); m.fit(tr[FEATS], tr.Presence)
imp = pd.DataFrame({'feature': FEATS,
                    'importance': m.feature_importances_}).sort_values(
    'importance', ascending=False)
imp.to_csv(os.path.join(RESULTS_DIR, f"feature_importance{suffix}.csv"), index=False)

print("\n" + "=" * 88)
print(f" FEATURE IMPORTANCE (all {len(FEATS)} features)")
print("=" * 88)
for _, r in imp.iterrows():
    src = ("Sentinel-1" if r.feature in ('RVI_Phenology_Delta', 'Moisture_T30',
                                         'Moisture_T15', 'RVI_T0')
           else "SMAP L-band" if r.feature.startswith('SMAP') else "ERA5")
    print(f"  {r.feature:<22}{r.importance * 100:>6.2f}%   {src}")

# ---------------------------------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------------------------------
# Gini importance is unsigned and biased towards continuous, high-variance
# features. SHAP gives a signed per-point contribution, and the two panels
# (fitted on the training points, applied to the test points) separate
# "uninformative" from "informative but non-transferring".
print("\n" + "=" * 88)
print(" SHAP EXPLAINABILITY -- signed feature contributions")
print("=" * 88)

try:
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tez_all = zscore(test_all, test_all, FEATS)
    te_fit = tez_all[tez_all.Week.isin(TEST_WEEKS)].dropna(subset=FEATS)

    def shap_matrix(explainer, X):
        """Return the positive-class SHAP matrix across shap versions.

        shap_values() returns a list of 2 arrays on older releases and a
        single (n, features, 2) array on newer ones. Both are handled.
        """
        sv = explainer.shap_values(X, check_additivity=False)
        if isinstance(sv, list):
            return np.asarray(sv[1])
        sv = np.asarray(sv)
        return sv[:, :, 1] if sv.ndim == 3 else sv

    # Beeswarm placement is cosmetic but stochastic. Current SHAP accepts an
    # explicit RNG; older releases used NumPy's global RNG, retained as a
    # compatibility fallback.
    shap_supports_rng = 'rng' in inspect.signature(shap.summary_plot).parameters
    shap_rng = np.random.default_rng(42)
    if not shap_supports_rng:
        np.random.seed(42)

    explainer = shap.TreeExplainer(m)
    panels = [("Fitted on hopper/band training points", tr[FEATS]),
              (f"Applied to April {TARGET} test points", te_fit[FEATS])]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, 0.5 * len(FEATS) + 3)))
    shap_rows = []
    for ax, (title, Xp) in zip(axes, panels):
        sv = shap_matrix(explainer, Xp)
        plt.sca(ax)
        plot_kwargs = {'rng': shap_rng} if shap_supports_rng else {}
        shap.summary_plot(sv, Xp, feature_names=FEATS, show=False,
                          plot_size=None, color_bar=(ax is axes[-1]),
                          **plot_kwargs)
        ax.set_title(f"{title}\n(n = {len(Xp)})", fontsize=11)
        ax.set_xlabel("SHAP value (impact on predicted swarm risk)", fontsize=9)

        # Direction: sign of corr(feature value, its own SHAP value).
        # Positive => higher values of the feature raise predicted risk.
        for j, f in enumerate(FEATS):
            x = Xp[f].values
            r = (np.corrcoef(x, sv[:, j])[0, 1]
                 if np.std(x) > 0 and np.std(sv[:, j]) > 0 else np.nan)
            shap_rows.append({'panel': title, 'feature': f,
                              'mean_abs_shap': round(float(np.abs(sv[:, j]).mean()), 5),
                              'direction_corr': round(float(r), 3)})

    fig.suptitle("SHAP feature impacts -- Random Forest, "
                 f"{len(FEATS)} features, seasonally corrected", fontsize=13)
    plt.tight_layout()
    shap_png = os.path.join(RESULTS_DIR, f"fig_shap_summary{suffix}.png")
    plt.savefig(shap_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

    sh = pd.DataFrame(shap_rows)
    sh.to_csv(os.path.join(RESULTS_DIR, f"shap_values{suffix}.csv"), index=False)

    print(f" {'feature':<22}{'|SHAP| train':>14}{'|SHAP| test':>14}"
          f"{'direction':>16}{'transfer':>11}")
    print(" " + "-" * 86)
    a, b = panels[0][0], panels[1][0]
    tr_s = sh[sh.panel == a].set_index('feature')
    te_s = sh[sh.panel == b].set_index('feature')
    for f in tr_s.mean_abs_shap.sort_values(ascending=False).index:
        d = te_s.loc[f, 'direction_corr']
        arrow = "higher->risk" if d > 0.05 else "lower->risk" if d < -0.05 else "mixed"
        ratio = (te_s.loc[f, 'mean_abs_shap'] / tr_s.loc[f, 'mean_abs_shap']
                 if tr_s.loc[f, 'mean_abs_shap'] else np.nan)
        print(f" {f:<22}{tr_s.loc[f, 'mean_abs_shap']:>14.4f}"
              f"{te_s.loc[f, 'mean_abs_shap']:>14.4f}{arrow:>16}{ratio:>10.2f}x")

    print("\n  'transfer' = |SHAP| on test / |SHAP| on train. Well below 1.0 means")
    print("  the feature stopped contributing once the target changed life stage.")
    print(f"  Saved {os.path.basename(shap_png)} and shap_values{suffix}.csv")

except ImportError:
    print("  [SKIPPED] shap is not installed.  pip install shap")
except Exception as e:                                    # noqa: BLE001
    # Never let an explainability plot kill the results run.
    print(f"  [SKIPPED] SHAP failed: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------
# DISTANCE BASELINE -- no satellite data at all
# ---------------------------------------------------------------------
print("\n" + "=" * 88)
print(" DISTANCE BASELINE -- ranks locations by proximity to a known hopper/band")
print(" site. Uses NO satellite data. Every AUC above must be read against this.")
print("=" * 88)
# Paired bootstrap: points are resampled once and both scorers re-evaluated
# on the same resample, so the interval is on the DIFFERENCE rather than on
# either AUC separately. 463 sightings occupy ~55 distinct ERA5 cells (09),
# so the effective sample is well below the nominal one.
N_BOOT = 2000
BOOT_RNG = np.random.RandomState(42)


def paired_bootstrap_auc_diff(y, s_a, s_b, n=N_BOOT):
    """95% CI on AUC(a) - AUC(b), resampling points and keeping pairs."""
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        b = BOOT_RNG.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        diffs.append(roc_auc_score(y[b], s_a[b]) - roc_auc_score(y[b], s_b[b]))
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


known = train_all[train_all.Week.isin(TRAIN_WEEKS) & (train_all.Presence == 1)]

# The headline model: ERA5 only, seasonally corrected -- the configuration
# quoted throughout. Rebuilt here explicitly rather than reused from the loop
# above, so this block cannot silently pick up whichever fit ran last.
tez_h = zscore(test_all, test_all, FEATS)
trh = trz[trz.Week.isin(TRAIN_WEEKS)].dropna(subset=ERA5_FEATURES)
m_head = make_model(); m_head.fit(trh[ERA5_FEATURES], trh.Presence)

base, mods = [], []
pool_y, pool_m, pool_d = [], [], []
print(f" {'week':<6}{'model AUC':>11}{'baseline AUC':>14}{'difference':>12}"
      f"{'95% CI on difference':>26}{'verdict':>18}")
print(" " + "-" * 86)
for wk in TEST_WEEKS:
    d = tez_h[tez_h.Week == wk].dropna(subset=ERA5_FEATURES)
    if len(d) == 0 or d.Presence.nunique() < 2:
        continue
    dist = np.array([haversine_km(r.Y, r.X, known.Y.values, known.X.values).min()
                     for _, r in d.iterrows()])
    proba = m_head.predict_proba(d[ERA5_FEATURES])[:, 1]
    y = d.Presence.values

    a_b = roc_auc_score(y, -dist)
    a_m = roc_auc_score(y, proba)
    diff, lo, hi = paired_bootstrap_auc_diff(y, proba, -dist)
    verdict = ("baseline wins" if hi < 0 else
               "model wins" if lo > 0 else "indistinguishable")
    base.append(a_b); mods.append(a_m)
    pool_y.append(y); pool_m.append(proba); pool_d.append(-dist)

    print(f" W{wk:<5}{a_m:>11.3f}{a_b:>14.3f}{a_m - a_b:>+12.3f}"
          f"{f'[{lo:+.3f}, {hi:+.3f}]':>26}{verdict:>18}")

y_all = np.concatenate(pool_y)
diff, lo, hi = paired_bootstrap_auc_diff(y_all, np.concatenate(pool_m),
                                         np.concatenate(pool_d))
pooled = ("baseline wins" if hi < 0 else
          "model wins" if lo > 0 else "indistinguishable")
best = out[out.seasonal_correction].groupby('features').auc.mean().max()

print(f"\n  Baseline mean AUC {np.mean(base):.3f}   vs   best model mean AUC {best:.3f}")
print(f"  POOLED W7-W10 (n={len(y_all)}): difference {diff:+.3f}, "
      f"95% CI [{lo:+.3f}, {hi:+.3f}]  ->  {pooled.upper()}")
if hi < 0:
    print("  The satellite model does not beat proximity on this data, and the")
    print("  interval excludes zero.")
elif lo > 0:
    print("  The model beats proximity and the interval excludes zero.")
else:
    print("  The interval spans zero: model and baseline are not statistically")
    print("  distinguishable on this data.")
pd.DataFrame([{'comparison': 'model_vs_distance_baseline', 'n': int(len(y_all)),
               'auc_model': round(float(np.mean(mods)), 4),
               'auc_baseline': round(float(np.mean(base)), 4),
               'auc_difference': round(diff, 4),
               'ci_low': round(lo, 4), 'ci_high': round(hi, 4),
               'verdict': pooled, 'n_bootstrap': N_BOOT}]).to_csv(
    os.path.join(RESULTS_DIR, f"baseline_comparison{suffix}.csv"), index=False)

# ---------------------------------------------------------------------
# DOES THE RANDOM FOREST EARN ITS COMPLEXITY?
# ---------------------------------------------------------------------
# Does the ensemble beat a linear model on the same features? Same weeks,
# same points, no tuning on either.
from sklearn.linear_model import LogisticRegression

print("\n" + "=" * 88)
print(" MODEL CLASS COMPARISON -- same 5 ERA5 features, same weeks")
print("=" * 88)
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(trh[ERA5_FEATURES], trh.Presence)
lr_auc = []
for wk in TEST_WEEKS:
    d = tez_h[tez_h.Week == wk].dropna(subset=ERA5_FEATURES)
    if len(d) == 0 or d.Presence.nunique() < 2:
        continue
    lr_auc.append(roc_auc_score(d.Presence, lr.predict_proba(d[ERA5_FEATURES])[:, 1]))
print(f"  {'Random Forest (100 trees, depth 5)':<40}{np.mean(mods):>8.3f}")
print(f"  {'Logistic regression (no tuning)':<40}{np.mean(lr_auc):>8.3f}")
print(f"  {'Distance baseline (no satellite data)':<40}{np.mean(base):>8.3f}")
gap = np.mean(mods) - np.mean(lr_auc)
print(f"\n  Forest minus logistic: {gap:+.3f}")
if abs(gap) < 0.02:
    print("  The forest does not beat a linear model on the same features.")
elif gap > 0:
    print("  The forest beats a linear model on the same features, so the")
    print("  feature-response relationships are non-linear. This concerns the model")
    print("  class only, and is independent of the baseline comparison.")
else:
    print("  A linear model beats the forest. The ensemble is overfitting.")

print(f"\n[SUCCESS] Saved results_weekly{suffix}.csv and feature_importance{suffix}.csv")
if not PRESENCE:
    print("Run again with --target presence to score all locust categories")
    print("(the target used by the published literature).")
print("Next: 05 -> 06 -> 07 -> 08 (risk maps), then 09 -> 10 -> 11 -> 12")
