# source_data/ — raw FAO field reports

Unmodified downloads from the FAO Locust Hub. These are **inputs**, not
results: nothing in this study produced them, and nothing here has been
filtered, cleaned or edited. Copies are kept inside this folder so the
pipeline runs standalone.

| File | Contents |
|---|---|
| `hoppers_2020.csv` | hopper sightings, 2020, global |
| `bands_2020.csv` | hopper band sightings, 2020, global |
| `swarms_2020.csv` | adult swarm sightings, 2020, global |
| `adults_2020.csv` | scattered adult sightings, 2020, global |
| `Desert locusts observation by day (Global).csv` | all observation categories by day, including `NO LOCUST` survey records used as real absences |
| `Desert Locust Bulletin 497 (5 March 2020).pdf` | contemporaneous FAO situation narrative; not consumed by the pipeline |

`01_prepare_points.py` filters these to the study ROI (36–39°E, 0–6°N) and
the 10-week label grid.

## Terms and attribution

These files are third-party material and are not covered by the repository's
MIT licence. FAO describes Locust Hub survey and control data as available for
research and other non-commercial purposes. Item-level terms and FAO publication
terms remain controlling; users should verify them for their intended reuse.
See the repository-level `NOTICE` for provenance, attribution and disclaimers.

## One parsing trap, documented because it caused a real error

The four `*_2020.csv` files are **ISO-dated**. The global observation file is
**DD-MM-YYYY**. Parsing the latter without `dayfirst=True` silently drops 59%
of in-ROI rows as `NaT` (any day > 12 fails month-first parsing) and
**misparses the rest** — `03-06-2020` (3 June) is read as 3 March.
`01_prepare_points.py` sets `dayfirst=True` for this file only.
