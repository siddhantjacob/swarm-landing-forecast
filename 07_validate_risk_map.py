"""
Scores the risk map against the real April swarm sightings.

Reports concentration = (share of swarms in a class) / (share of area in
it), where 1.0 means no better than surveying that much ground at random;
and the number of bases needed to service the alert within a given radius.
The latter is a k-center problem, solved by farthest-point clustering
(Gonzalez 1985, Theoretical Computer Science 38:293-306), which is
deterministic and monotone in the number of bases.

Precision from the point-based test does not transfer here: that dataset is
roughly balanced, whereas swarms are rare across 220,000 km2. Recall and
concentration do transfer.

FAO sightings are survey-effort biased -- an unreported pixel means nobody
looked, not that nothing landed -- so every figure is an upper bound.

OUTPUT: risk_map_validation.csv
"""

import numpy as np
import pandas as pd
import rasterio
import os
import json

from timeline_config import RESULTS_DIR, DATA_DIR, BASE_DIR, week_bounds, test_lead_days

RASTER_DIR = os.path.join(DATA_DIR, "risk_rasters")
OPER = os.path.join(RASTER_DIR, "risk_operational.tif")
META = os.path.join(RASTER_DIR, "risk_map_thresholds.json")
TEST_WEEKS = [7, 8, 9, 10]

NAMES = {4: 'Known infested area', 3: 'HIGH', 2: 'MEDIUM', 1: 'LOW', 0: 'Background'}

print("[START] 07 - Validating the risk map against real April swarms\n")
for p in (OPER, META):
    if not os.path.exists(p):
        print(f"[FATAL] {p} not found. Run 06_generate_risk_map.py first.")
        raise SystemExit(1)

with open(META) as fh:
    meta = json.load(fh)
counts = {int(k): v for k, v in meta['operational_class_pixels'].items()}
total = sum(counts.values())
px_km2 = meta['grid']['px_km2']
area = {k: v / total for k, v in counts.items()}

pts = pd.read_csv(os.path.join(DATA_DIR, "all_points.csv"))
sw = pts[(pts.Presence == 1) & (pts.Source == 'fao_swarm') &
         (pts.Week.isin(TEST_WEEKS))].copy()
with rasterio.open(OPER) as src:
    sw['cls'] = [v[0] for v in src.sample(list(zip(sw.X, sw.Y)))]
off = int((sw.cls == 255).sum())
sw = sw[sw.cls != 255]
print(f" -> {len(sw)} swarm sightings on the map ({off} fell outside the habitat mask)\n")

# ---------------------------------------------------------------------
print("=" * 82)
print(" DO SWARMS CONCENTRATE WHERE THE MAP SAYS THEY WILL?")
print("=" * 82)
print(f"  {'class':<22}{'swarms':>8}{'% swarms':>11}{'% area':>9}"
      f"{'CONCENTRATION':>15}{'area km2':>12}")
print("  " + "-" * 78)
rows = []
for k in (4, 3, 2, 1, 0):
    n = int((sw.cls == k).sum())
    ss, sa = n / len(sw), area.get(k, 0.0)
    conc = ss / sa if sa else np.nan
    ct = f"{conc:>14.2f}x" if np.isfinite(conc) else f"{'n/a':>15}"
    print(f"  {NAMES[k]:<22}{n:>8}{ss:>10.1%}{sa:>9.1%}{ct}{counts.get(k,0)*px_km2:>12,.0f}")
    rows.append({'class': k, 'class_name': NAMES[k], 'swarms': n,
                 'share_swarms': round(ss, 4), 'share_area': round(sa, 4),
                 'concentration': round(conc, 3) if np.isfinite(conc) else None,
                 'area_km2': round(counts.get(k, 0) * px_km2, 1)})

# ---------------------------------------------------------------------
print("\n" + "=" * 82)
print(" WEEK BY WEEK -- swarms falling in an alerted area")
print("=" * 82)
print(f"  {'week':<6}{'dates':<24}{'lead':>6}{'swarms':>8}{'alerted':>9}{'missed':>8}{'recall':>9}")
print("  " + "-" * 70)
for wk in TEST_WEEKS:
    d = sw[sw.Week == wk]
    if not len(d):
        continue
    hit = int(d.cls.isin([2, 3, 4]).sum())
    s, e = week_bounds(wk)
    print(f"  W{wk:<5}{str(s.date())+'..'+str(e.date()):<24}"
          f"{str(test_lead_days(wk))+'d':>6}{len(d):>8}{hit:>9}{len(d)-hit:>8}{hit/len(d):>9.1%}")
    rows.append({'class': 'weekly', 'class_name': f'W{wk}', 'swarms': len(d),
                 'share_swarms': round(hit / len(d), 4), 'share_area': None,
                 'concentration': None, 'area_km2': None})

alert_area = sum(area.get(k, 0.0) for k in (2, 3, 4))
alert_hit = int(sw.cls.isin([2, 3, 4]).sum())
print(f"\n  Alerted area = known infested + HIGH + MEDIUM = {alert_area:.1%} of the region "
      f"({alert_area*total*px_km2:,.0f} km2)")
print(f"  It contains {alert_hit}/{len(sw)} ({alert_hit/len(sw):.1%}) of the April swarms "
      f"-- {(alert_hit/len(sw))/alert_area:.2f}x the density of surveying at random.")

# ---------------------------------------------------------------------
# OPERATIONAL COST -- what would acting on this alert actually take?
# ---------------------------------------------------------------------
# A percentage of area is not a decision. A control programme is limited by
# how many teams it can field and how far each can travel from its base, so
# the alert is converted into that unit: the smallest number of bases such
# that every alerted pixel lies within a given service radius of one.
#
# The service radius is NOT tuned or justified from the data -- there is no
# defensible single value -- so a range is reported instead and the reader
# applies whichever matches their fleet. What the design does control for is
# the comparison: bases needed for the KNOWN INFESTED AREA ALONE (which
# needs no model at all, only FAO reports) versus bases needed for the FULL
# ALERT. The difference is the marginal cost of acting on the model, and it
# is set against the marginal swarms the model adds. That ratio is the
# operational version of the concentration figures above.
#
# Fixed-radius coverage, approximated by deterministic farthest-first
# clustering. Gonzalez's 2-approximation guarantee is for the reverse
# fixed-k problem (minimising the maximum radius), so these counts are
# comparative estimates rather than globally optimal base counts.
SERVICE_RADII_KM = (25, 50, 75)
MAX_BASES = 400
# Alerted pixels are subsampled before clustering. At ~8,000 samples over
# ~46,000 km2 the mean spacing is about 2.4 km, well inside the smallest
# service radius tested, so the base counts are not sensitive to this.
# It does mean the max-distance criterion is evaluated on a sample, which
# biases the base count very slightly LOW -- i.e. optimistic.
MAX_PIXELS = 8000
RNG = np.random.RandomState(42)

print("\n" + "=" * 82)
print(" OPERATIONAL COST -- bases needed to service the alert")
print("=" * 82)

with rasterio.open(OPER) as src:
    band = src.read(1)
    rr, cc = np.nonzero(np.isin(band, [2, 3, 4]))
    xs, ys = rasterio.transform.xy(src.transform, rr, cc)
cls_at = band[rr, cc]
xs, ys = np.asarray(xs), np.asarray(ys)

# Project to local kilometres so clustering is not distorted by latitude.
lat0 = float(np.mean(ys))
to_km = lambda x, y: np.column_stack([x * 111.32 * np.cos(np.radians(lat0)), y * 111.32])


def bases_needed(P, radius_km):
    """Bases so every pixel is within radius_km of one (Gonzalez 1985).

    Seeds at the pixel farthest from the centroid -- an arbitrary but FIXED
    choice, so the result is reproducible -- then repeatedly opens a base at
    whichever pixel is currently worst served. `d` holds each pixel's
    distance to its nearest base and falls monotonically, so the loop always
    terminates and the count never decreases when the region grows.
    """
    if len(P) == 0:
        return None, None
    seed = int(np.argmax(((P - P.mean(0)) ** 2).sum(1)))
    d = np.sqrt(((P - P[seed]) ** 2).sum(1))
    n = 1
    while d.max() > radius_km and n < MAX_BASES:
        nxt = int(np.argmax(d))
        d = np.minimum(d, np.sqrt(((P - P[nxt]) ** 2).sum(1)))
        n += 1
    return (n, float(d.max())) if d.max() <= radius_km else (None, float(d.max()))


regions = {
    'known infested only (no model)': cls_at == 4,
    'full alert (known + HIGH + MEDIUM)': np.isin(cls_at, [2, 3, 4]),
}
sw_in = {'known infested only (no model)': int((sw.cls == 4).sum()),
         'full alert (known + HIGH + MEDIUM)': int(sw.cls.isin([2, 3, 4]).sum())}

sampled = {}
for name, mask in regions.items():
    idx = np.nonzero(mask)[0]
    if len(idx) > MAX_PIXELS:
        idx = RNG.choice(idx, MAX_PIXELS, replace=False)
    sampled[name] = to_km(xs[idx], ys[idx])
    print(f"  {name:<38}{mask.sum() * px_km2:>10,.0f} km2   "
          f"{sw_in[name]:>4} swarms")

cap = MAX_BASES
print(f"\n  {'service radius':<18}{'known infested':>18}{'full alert':>14}"
      f"{'extra bases':>14}{'extra per swarm':>18}")
print("  " + "-" * 78)
for radius in SERVICE_RADII_KM:
    got = {}
    for name in regions:
        k, worst = bases_needed(sampled[name], radius)
        got[name] = k
    a = got['known infested only (no model)']
    b = got['full alert (known + HIGH + MEDIUM)']
    extra = (b - a) if (a is not None and b is not None) else None
    d_sw = sw_in['full alert (known + HIGH + MEDIUM)'] - sw_in['known infested only (no model)']
    per = f"{extra / d_sw:.1f}" if (extra is not None and d_sw > 0) else "n/a"
    fmt = lambda k: (str(k) if k else f">{cap}")
    print(f"  {str(radius) + ' km':<18}{fmt(a):>18}{fmt(b):>14}"
          f"{(f'+{extra}' if extra is not None else 'n/a'):>14}{per:>18}")
    rows.append({'class': 'ops_cost', 'class_name': f'service_radius_{radius}km',
                 'swarms': None, 'share_swarms': None, 'share_area': None,
                 'concentration': None, 'area_km2': None,
                 'bases_known_only': a, 'bases_full_alert': b, 'extra_bases': extra})

extra_swarms = sw_in['full alert (known + HIGH + MEDIUM)'] - sw_in['known infested only (no model)']
extra_area = (np.isin(cls_at, [2, 3]).sum()) * px_km2
print(f"\n  Going from 'survey the known infested area' to 'survey the full alert'")
print(f"  adds {extra_area:,.0f} km2 of ground and finds {extra_swarms} additional swarms")
print(f"  ({extra_swarms / len(sw):.1%} of the total).")
print("\n  This is the operational form of the concentration table above:")
print("  alert recall set against the survey capacity it would require.")

pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "risk_map_validation.csv"), index=False)
print(f"\n[SUCCESS] Saved risk_map_validation.csv")
print("Next: run 08_make_figures.py")
