"""
Renders the risk-map figures and the QGIS deliverables.

OUTPUT: figures/ (4 png); .qml styles and overlay CSVs into
        Data/risk_rasters/
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from timeline_config import (RESULTS_DIR, DATA_DIR, BASE_DIR, ROI_COORDS, REANCHOR_TRAIN_WEEKS,
                             REANCHOR_TEST_WEEKS, TEST_ANCHOR_DATE,
                             week_bounds, test_lead_days)

RASTER_DIR = os.path.join(DATA_DIR, "risk_rasters")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

OPER = os.path.join(RASTER_DIR, "risk_operational.tif")
META = os.path.join(RASTER_DIR, "risk_map_thresholds.json")

print("[START] 08 - Rendering risk-map figures...")
for p in (OPER, META):
    if not os.path.exists(p):
        print(f"[FATAL] {p} not found. Run 06_generate_risk_map.py first.")
        raise SystemExit(1)

with open(META) as fh:
    meta = json.load(fh)

with rasterio.open(OPER) as src:
    grid = src.read(1)
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# ---------------------------------------------------------------------
# POINTS
# ---------------------------------------------------------------------
tr = pd.read_csv(os.path.join(DATA_DIR, "train_pool", "features_all.csv"))
bands = tr[tr.Week.isin(REANCHOR_TRAIN_WEEKS) & (tr.Presence == 1)]

pts = pd.read_csv(os.path.join(DATA_DIR, "all_points.csv"))
sw = pts[(pts.Presence == 1) & (pts.Source == 'fao_swarm') &
         (pts.Week.isin(REANCHOR_TEST_WEEKS))]
print(f" -> {len(bands)} training band sites, {len(sw)} April swarm sightings")

# ---------------------------------------------------------------------
# COLOURS -- buffer deliberately OUTSIDE the risk ramp (see docstring)
# ---------------------------------------------------------------------
COLORS = {
    0: "#f2f0eb",   # background
    1: "#ffe08a",   # LOW
    2: "#f79a3e",   # MEDIUM
    3: "#c1272d",   # HIGH
    4: "#9aa5b1",   # known buffer -- grey, not on the ramp
}
cmap = ListedColormap([COLORS[k] for k in (0, 1, 2, 3, 4)])
cmap.set_bad("#ffffff")
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
plot_grid = np.ma.masked_where(grid == 255, grid)

LEGEND = [
    Patch(facecolor=COLORS[4], label='Known activity, ≤30 km (FAO reports, no model)'),
    Patch(facecolor=COLORS[3], label='HIGH — model, top 5%'),
    Patch(facecolor=COLORS[2], label='MEDIUM — model, top 10%'),
    Patch(facecolor=COLORS[1], label='LOW — model, top 25%'),
    Patch(facecolor=COLORS[0], label='Background'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
           markeredgecolor='#111111', markersize=6, label='April swarm sighting'),
    Line2D([0], [0], marker='^', color='none', markerfacecolor='#1b5e20',
           markeredgecolor='none', markersize=7, label='Training hopper/band site'),
]

# =====================================================================
# FIGURE 1 -- overview
# =====================================================================
fig, ax = plt.subplots(figsize=(8.2, 11.4))
ax.imshow(plot_grid, cmap=cmap, norm=norm, extent=extent, origin='upper',
          interpolation='nearest')
ax.scatter(bands.X, bands.Y, marker='^', s=26, c='#1b5e20',
           edgecolors='none', zorder=3, alpha=0.9)
ax.scatter(sw.X, sw.Y, marker='o', s=17, facecolors='none',
           edgecolors='#111111', linewidths=0.6, zorder=4, alpha=0.85)

ax.set_xlim(ROI_COORDS[0], ROI_COORDS[2])
ax.set_ylim(ROI_COORDS[1], ROI_COORDS[3])
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(
    f"Desert locust swarm risk, {week_bounds(REANCHOR_TEST_WEEKS[0])[0].date()} – "
    f"{week_bounds(REANCHOR_TEST_WEEKS[-1])[1].date()}\n"
    f"ERA5-only model, anchored {TEST_ANCHOR_DATE.date()}, "
    f"{test_lead_days(REANCHOR_TEST_WEEKS[0])}–{test_lead_days(REANCHOR_TEST_WEEKS[-1])} day lead",
    fontsize=11, pad=12)
ax.legend(handles=LEGEND, loc='upper left', fontsize=7.5, framealpha=0.94)
ax.text(0.5, -0.062,
        "ERA5-Land native resolution ≈11 km; pixels shown at 1.1 km are interpolated, not detail.\n"
        "The grey buffer is FAO field reporting, not a model output — it alone contains 82.7% of "
        "these swarms.",
        transform=ax.transAxes, ha='center', va='top', fontsize=7.2, color='#444444')
fig.tight_layout()
p1 = os.path.join(FIG_DIR, "fig_risk_map_overview.png")
fig.savefig(p1, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"    wrote {os.path.basename(p1)}")

# =====================================================================
# FIGURE 2 -- weekly panel
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(11.0, 13.0))
for ax, wk in zip(axes.ravel(), REANCHOR_TEST_WEEKS):
    d = sw[sw.Week == wk]
    s, e = week_bounds(wk)
    ax.imshow(plot_grid, cmap=cmap, norm=norm, extent=extent, origin='upper',
              interpolation='nearest')
    ax.scatter(bands.X, bands.Y, marker='^', s=14, c='#1b5e20',
               edgecolors='none', zorder=3, alpha=0.75)
    ax.scatter(d.X, d.Y, marker='o', s=20, facecolors='none',
               edgecolors='#111111', linewidths=0.7, zorder=4)
    ax.set_xlim(ROI_COORDS[0], ROI_COORDS[2])
    ax.set_ylim(ROI_COORDS[1], ROI_COORDS[3])
    ax.set_title(f"W{wk}  {s.date()} – {e.date()}\n"
                 f"{len(d)} swarms, {test_lead_days(wk)} d lead", fontsize=9.5)
    ax.tick_params(labelsize=7)
axes[0, 0].legend(handles=LEGEND[:5], loc='upper left', fontsize=6.5, framealpha=0.94)
fig.suptitle("Weekly swarm risk maps — identical model, one week at a time", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.985])
p2 = os.path.join(FIG_DIR, "fig_risk_map_weekly.png")
fig.savefig(p2, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"    wrote {os.path.basename(p2)}")

# =====================================================================
# FIGURE 3 -- concentration: swarms caught vs area covered
# =====================================================================
val_path = os.path.join(RESULTS_DIR, "risk_map_validation.csv")
if os.path.exists(val_path):
    v = pd.read_csv(val_path)
    # 07 writes one row per risk class (4=known infested, 3/2/1/0=model) plus
    # weekly rows tagged 'weekly'. Take the class rows only.
    v['class_num'] = pd.to_numeric(v['class'], errors='coerce')
    cls = v.dropna(subset=['class_num']).set_index(v.dropna(subset=['class_num'])
                                                   .class_num.astype(int))

    # cumulative curve over the MODEL classes, best first
    cum_a, cum_s, labels = [], [], []
    a_ = s_ = 0.0
    for k, lbl in ((3, 'HIGH'), (2, '+MEDIUM'), (1, '+LOW')):
        if k not in cls.index:
            continue
        a_ += float(cls.loc[k, 'share_area'])
        s_ += float(cls.loc[k, 'share_swarms'])
        cum_a.append(a_); cum_s.append(s_); labels.append(lbl)

    buf_a = float(cls.loc[4, 'share_area']) if 4 in cls.index else np.nan
    buf_s = float(cls.loc[4, 'share_swarms']) if 4 in cls.index else np.nan

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.plot([0, 1], [0, 1], ls='--', c='#888888', lw=1.2, label='No skill (1:1)')
    if cum_a:
        ax.plot(cum_a, cum_s, marker='o', c='#c1272d', lw=1.8,
                label='Satellite model risk classes')
        OFF = {'HIGH': (12, 10), '+MEDIUM': (14, -20), '+LOW': (14, -4)}
        for x, y, t in zip(cum_a, cum_s, labels):
            ax.annotate(f"{t}\n{y:.0%} of swarms in {x:.0%} of area",
                        (x, y), textcoords="offset points", xytext=OFF.get(t, (10, -4)),
                        fontsize=7.6, color='#333333',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.8))
    if np.isfinite(buf_a):
        ax.scatter([buf_a], [buf_s], marker='s', s=90, c='#9aa5b1',
                   edgecolors='#555555', zorder=5,
                   label='Known infested area (no model)')
        ax.annotate(f"FAO field reports + 30 km — no model\n"
                    f"{buf_s:.0%} of swarms in {buf_a:.0%} of area  ({buf_s/buf_a:.2f}x)",
                    (buf_a, buf_s), textcoords="offset points", xytext=(14, 4),
                    fontsize=8.0, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbbbbb', alpha=0.9))

    ax.set_xlabel("Share of habitat area flagged")
    ax.set_ylabel("Share of swarms caught")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("What each level of recall costs in ground to survey\n"
                 "Above the diagonal is better than random; the grey square uses no satellite data",
                 fontsize=10.5)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    p3 = os.path.join(FIG_DIR, "fig_concentration.png")
    fig.savefig(p3, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {os.path.basename(p3)}")
else:
    print(f"    [SKIP] {val_path} not found -- run 07_validate_risk_map.py for figure 3.")

print(f"\n[SUCCESS] Figures in {FIG_DIR}")
print("  Use fig_risk_map_overview.png as the main map figure and")
print("  fig_concentration.png as the performance figure. The second is the more")
print("  honest of the two -- it shows the area cost that a coloured map hides.")


# =====================================================================
# QGIS STYLES -- so the rasters open in colour, not near-black greyscale.
# The rasters are uint8 0-4; QGIS stretches greyscale over 0-255 by
# default. A .qml beside the .tif is picked up automatically on load.
# =====================================================================
def qml(entries):
    rows = "\n".join(
        f'          <paletteEntry value="{v}" alpha="{a}" color="{c}" label="{l}"/>'
        for v, c, a, l in entries)
    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.28" styleCategories="AllStyleCategories">\n'
        "  <pipe>\n"
        '    <rasterrenderer band="1" type="paletted" opacity="1" '
        'nodataColor="" alphaBand="-1">\n'
        "      <rasterTransparency/>\n"
        "      <colorPalette>\n" + rows + "\n"
        "      </colorPalette>\n"
        "    </rasterrenderer>\n"
        '    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>\n'
        '    <huesaturation colorizeOn="0" saturation="0" grayscaleMode="0"/>\n'
        '    <rasterresampler maxOversampling="2"/>\n'
        "  </pipe>\n"
        "  <blendMode>0</blendMode>\n"
        "</qgis>\n")


STYLES = {
    "risk_operational": [
        (0, "#f2f0eb", 255, "Background"),
        (1, "#ffe08a", 255, "LOW - model, top 25%"),
        (2, "#f79a3e", 255, "MEDIUM - model, top 10%"),
        (3, "#c1272d", 255, "HIGH - model, top 5%"),
        (4, "#9aa5b1", 255, "Known activity &lt;=30km (FAO reports, no model)"),
    ],
    "risk_model_classes": [
        (0, "#f2f0eb", 255, "Background"),
        (1, "#ffe08a", 255, "LOW - top 25%"),
        (2, "#f79a3e", 255, "MEDIUM - top 10%"),
        (3, "#c1272d", 255, "HIGH - top 5%"),
    ],
    "risk_known_buffer": [
        (0, "#f2f0eb", 0, "Outside buffer"),
        (1, "#9aa5b1", 255, "Within 30 km of known band activity"),
    ],
}

for name, ents in STYLES.items():
    tif = os.path.join(RASTER_DIR, name + ".tif")
    if not os.path.exists(tif):
        print(f"    [SKIP] {name}.tif not found -- run 06_generate_risk_map.py first.")
        continue
    with open(os.path.join(RASTER_DIR, name + ".qml"), "w", encoding="utf-8") as fh:
        fh.write(qml(ents))
    print(f"    wrote {name}.qml")

# --- overlays ---
tp = os.path.join(DATA_DIR, "train_pool", "features_all.csv")
if os.path.exists(tp):
    tr = pd.read_csv(tp)
    b = tr[tr.Week.isin(REANCHOR_TRAIN_WEEKS) & (tr.Presence == 1)][['X', 'Y', 'Week', 'Source']]
    b.to_csv(os.path.join(RASTER_DIR, "overlay_training_band_sites.csv"), index=False)
    print(f"    wrote overlay_training_band_sites.csv ({len(b)} points)")

ap = os.path.join(DATA_DIR, "all_points.csv")
if os.path.exists(ap):
    p = pd.read_csv(ap)
    s = p[(p.Presence == 1) & (p.Source == 'fao_swarm') &
          (p.Week.isin(REANCHOR_TEST_WEEKS))][['X', 'Y', 'Week', 'WeekStart', 'ObsDate']]
    s.to_csv(os.path.join(RASTER_DIR, "overlay_april_swarms.csv"), index=False)
    print(f"    wrote overlay_april_swarms.csv ({len(s)} points)")

print(f"\n[SUCCESS] Open {os.path.join(RASTER_DIR, 'risk_operational.tif')} in QGIS.")
print("  The .qml alongside it loads automatically. Add the overlay CSVs via")
print("  Layer > Add Layer > Add Delimited Text Layer (X=X, Y=Y, EPSG:4326).")


# =====================================================================
# FIGURE 4 -- THE COMPARISON THE STUDY RESTS ON
# Model performance against the distance-only baseline, week by week.
# This is the figure to put in the Results section: it shows in one glance
# that the satellite model does not beat proximity to known infestations.
# =====================================================================
def figure_model_vs_baseline():
    res_path = os.path.join(RESULTS_DIR, "results_weekly.csv")
    if not os.path.exists(res_path):
        print("    [SKIP] results_weekly.csv not found -- run 04_train_and_test.py.")
        return

    d = pd.read_csv(res_path)
    era5 = d[(d.seasonal_correction) & (d.features == 'era5')].sort_values('week')
    allf = d[(d.seasonal_correction) & (d.features == 'all')].sort_values('week')
    raw = d[(~d.seasonal_correction) & (d.features == 'era5')].sort_values('week')

    # Baseline AUCs are printed by 04 but not saved; recompute them here so the
    # figure cannot drift out of step with the table.
    tr = pd.read_csv(os.path.join(DATA_DIR, "train_pool", "features_all.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test_anchor", "features_all.csv"))
    te = te[(te.Presence == 0) | (te.Source == 'fao_swarm')]
    known = tr[tr.Week.isin(REANCHOR_TRAIN_WEEKS) & (tr.Presence == 1)]
    from sklearn.metrics import roc_auc_score
    base = []
    for wk in REANCHOR_TEST_WEEKS:
        w = te[te.Week == wk].dropna(subset=ERA5_FEATURES)
        dist = np.array([haversine_km(r.Y, r.X, known.Y.values, known.X.values).min()
                         for _, r in w.iterrows()])
        base.append(roc_auc_score(w.Presence, -dist))

    weeks = [f"W{w}\n{int(l)} d" for w, l in zip(era5.week, era5.lead_days)]
    x = np.arange(len(weeks))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4))

    # --- left: AUC ---
    ax1.plot(x, base, 'o-', c='#333333', lw=2.2, ms=8, zorder=5,
             label='Distance to known infestation (no satellite data)')
    ax1.plot(x, era5.auc, 's-', c='#c1272d', lw=2, ms=7,
             label='ERA5 weather (5 features)')
    ax1.plot(x, allf.auc, '^--', c='#f79a3e', lw=1.8, ms=7,
             label='ERA5 + Sentinel-1 (9 features)')
    ax1.plot(x, raw.auc, 'v:', c='#9aa5b1', lw=1.6, ms=6,
             label='ERA5, no seasonal correction')
    ax1.axhline(0.5, ls='--', c='#cccccc', lw=1)
    ax1.text(len(x) - 0.55, 0.512, 'chance', fontsize=7.5, color='#999999', ha='right')
    ax1.set_xticks(x); ax1.set_xticklabels(weeks, fontsize=9)
    ax1.set_ylim(0.45, 0.95); ax1.set_ylabel("AUC", fontsize=10)
    ax1.set_title("Discrimination", fontsize=11)
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.legend(fontsize=8, loc='lower left', framealpha=0.95)

    # --- right: recall ---
    w = 0.26
    ax2.bar(x - w, era5.recall, w, color='#c1272d', label='ERA5 weather')
    ax2.bar(x, allf.recall, w, color='#f79a3e', label='ERA5 + Sentinel-1')
    ax2.bar(x + w, raw.recall, w, color='#9aa5b1', label='No seasonal correction')
    for i, v in enumerate(era5.recall):
        ax2.text(i - w, v + 0.015, f"{v:.0%}", ha='center', fontsize=7.5, color='#c1272d')
    ax2.set_xticks(x); ax2.set_xticklabels(weeks, fontsize=9)
    ax2.set_ylim(0, 0.95)
    ax2.set_ylabel("Recall — share of swarms caught", fontsize=10)
    ax2.set_title("Detection at a fixed 0.5 threshold", fontsize=11)
    ax2.grid(alpha=0.25, lw=0.6, axis='y')
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.95)

    fig.suptitle("The satellite model does not outperform proximity to known infestations",
                 fontsize=12.5)
    fig.text(0.5, -0.02,
             f"Mean AUC — baseline {np.mean(base):.3f}, ERA5 {era5.auc.mean():.3f}, "
             f"+Sentinel-1 {allf.auc.mean():.3f}, uncorrected {raw.auc.mean():.3f}.   "
             "Same model, same weeks, threshold 0.5, nothing tuned.",
             ha='center', fontsize=8.5, color='#444444')
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    p = os.path.join(FIG_DIR, "fig_model_vs_baseline.png")
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    wrote {os.path.basename(p)}")


from timeline_config import REANCHOR_TRAIN_WEEKS, REANCHOR_TEST_WEEKS, ERA5_FEATURES, haversine_km
figure_model_vs_baseline()
