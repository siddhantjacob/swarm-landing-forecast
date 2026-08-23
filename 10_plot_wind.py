"""
Plots the ERA5-Land 10 m wind field over the training period: one panel per
training week plus the combined mean, with that week's hopper and band
sightings overlaid.

Descriptive only. Wind is not a model feature and nothing downstream reads
this output.

Arrows point the direction the wind is blowing, which is the direction a
fledged adult would drift. Meteorological convention names wind by where it
comes from, the opposite; both bearings are written to the CSV. Bearings are
averaged as vectors rather than as angles.

ERA5-Land wind is a 10 m, ~11 km, daily-mean field. Swarms fly higher and
only in daylight, so a mean arrow is a prevailing tendency over a week, not
a flight path.

OUTPUT: fig_wind_training_period.png, wind_training_period.csv
        REQUIRES GEE AUTH
"""

import ee
import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from timeline_config import (RESULTS_DIR, BASE_DIR, DATA_DIR, ROI_COORDS,
                             week_bounds, require_gee_project)

TRAIN_WEEKS = [1, 2, 3, 4, 5]
GRID_DEG = 0.25            # arrow spacing; ERA5-Land native is ~0.1 deg
ERA5_SCALE = 11132

print("[START] 10 - Plotting the wind field over the training period\n")
GEE_PROJECT = require_gee_project()

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    print("[AUTH NEEDED] Opening browser to authenticate Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

roi = ee.Geometry.Rectangle(ROI_COORDS)
ERA5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')

# ---------------------------------------------------------------------
# Sampling grid
# ---------------------------------------------------------------------
lons = np.arange(ROI_COORDS[0] + GRID_DEG / 2, ROI_COORDS[2], GRID_DEG)
lats = np.arange(ROI_COORDS[1] + GRID_DEG / 2, ROI_COORDS[3], GRID_DEG)
grid = [(float(x), float(y)) for y in lats for x in lons]
print(f" -> {len(lons)} x {len(lats)} = {len(grid)} grid points at {GRID_DEG} deg")

# ---------------------------------------------------------------------
# One multi-band image: u and v for every panel, so this is a SINGLE
# server call rather than six.
# ---------------------------------------------------------------------
panels = [(f"W{wk}", *week_bounds(wk)) for wk in TRAIN_WEEKS]
p_start, p_end = week_bounds(TRAIN_WEEKS[0])[0], week_bounds(TRAIN_WEEKS[-1])[1]
panels.append(("ALL", p_start, p_end))

bands = []
for tag, s, e in panels:
    c = ERA5.filterBounds(roi).filterDate(str(s.date()), str(e.date()))
    bands.append(c.select('u_component_of_wind_10m').mean().rename(f'U_{tag}'))
    bands.append(c.select('v_component_of_wind_10m').mean().rename(f'V_{tag}'))
combo = ee.Image.cat(bands)

fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([x, y]), {'i': i})
                           for i, (x, y) in enumerate(grid)])
print(" -> sampling ERA5-Land 10 m wind...")
got = combo.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                          scale=ERA5_SCALE).getInfo()

vals = {}
for f in got['features']:
    p = f['properties']
    vals[p['i']] = p
gx = np.array([g[0] for g in grid])
gy = np.array([g[1] for g in grid])


def uv(tag):
    u = np.array([vals.get(i, {}).get(f'U_{tag}', np.nan) for i in range(len(grid))], float)
    v = np.array([vals.get(i, {}).get(f'V_{tag}', np.nan) for i in range(len(grid))], float)
    return u, v


# ---------------------------------------------------------------------
# Sightings to overlay
# ---------------------------------------------------------------------
tp = os.path.join(DATA_DIR, "train_points.csv")
pts = pd.read_csv(tp) if os.path.exists(tp) else pd.DataFrame(columns=['X', 'Y', 'Presence', 'Week'])
sight = pts[pts.Presence == 1] if len(pts) else pts
if not len(sight):
    print(" -> [WARN] Data/train_points.csv not found; plotting wind only")


def compass(bearing):
    pts16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
             'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return pts16[int((bearing % 360) / 22.5 + 0.5) % 16]


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 13))
rows = []
print(f"\n {'panel':<16}{'dates':<26}{'speed':>8}{'blowing TOWARD':>20}{'coming FROM':>18}")
print(" " + "-" * 86)

for ax, (tag, s, e) in zip(axes.ravel(), panels):
    u, v = uv(tag)
    ok = np.isfinite(u) & np.isfinite(v)
    spd = np.sqrt(u ** 2 + v ** 2)
    # Direction the wind BLOWS TOWARD -- the direction an insect drifts.
    toward = (np.degrees(np.arctan2(u, v)) + 360) % 360
    m_spd = float(np.nanmean(spd))
    # Vector mean, not the mean of the angles: averaging bearings directly
    # is wrong when they straddle 0/360.
    m_toward = float((np.degrees(np.arctan2(np.nanmean(u), np.nanmean(v))) + 360) % 360)
    m_from = (m_toward + 180) % 360

    q = ax.quiver(gx[ok], gy[ok], u[ok], v[ok], spd[ok], cmap='viridis',
                  scale=45, width=0.005, pivot='mid')
    if len(sight):
        d = sight if tag == "ALL" else sight[sight.Week == int(tag[1:])]
        if len(d):
            ax.scatter(d.X, d.Y, s=16, c='crimson', edgecolor='white',
                       linewidth=0.4, zorder=3,
                       label=f'{len(d)} hopper/band sightings')
            ax.legend(loc='lower left', fontsize=7, framealpha=0.9)

    label = "W1-W5 combined" if tag == "ALL" else tag
    ax.set_title(f"{label}   {s.date()} .. {e.date()}\n"
                 f"mean {m_spd:.2f} m/s, blowing toward {m_toward:.0f}° "
                 f"({compass(m_toward)})", fontsize=10)
    ax.set_xlim(ROI_COORDS[0], ROI_COORDS[2])
    ax.set_ylim(ROI_COORDS[1], ROI_COORDS[3])
    ax.set_aspect('equal')
    ax.set_xlabel('longitude (°E)', fontsize=8)
    ax.set_ylabel('latitude (°N)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linestyle=':')
    plt.colorbar(q, ax=ax, fraction=0.035, pad=0.02, label='m/s')

    print(f" {label:<16}{str(s.date()) + '..' + str(e.date()):<26}{m_spd:>7.2f}m"
          f"{f'{m_toward:.0f} deg ({compass(m_toward)})':>20}"
          f"{f'{m_from:.0f} deg ({compass(m_from)})':>18}")
    rows.append({'panel': label, 'start': str(s.date()), 'end': str(e.date()),
                 'mean_speed_ms': round(m_spd, 3),
                 'blowing_toward_deg': round(m_toward, 1),
                 'blowing_toward': compass(m_toward),
                 'coming_from_deg': round(m_from, 1),
                 'coming_from': compass(m_from)})

fig.suptitle("ERA5-Land 10 m wind over the training period\n"
             "arrows point the way the wind is blowing, i.e. the way a fledged "
             "swarm would drift; red dots are hopper/band sightings",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
png = os.path.join(RESULTS_DIR, "fig_wind_training_period.png")
plt.savefig(png, dpi=200, bbox_inches='tight')
plt.close(fig)

pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "wind_training_period.csv"),
                          index=False)
print(f"\n[SUCCESS] Saved {os.path.basename(png)} and wind_training_period.csv")
print("  Descriptive figure only -- wind is not a model feature and nothing")
print("  downstream depends on this script.")
