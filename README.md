# ROVDataConcat

Processing pipeline for ROV (Hercules / Atalanta) expedition data: extracts
navigation, orientation, USBL, and sensor data from raw expedition files,
merges them onto a common 1 Hz UTC timeline, applies a Kalman filter, and
produces a final datatable (plus terrain-offset positions for Unreal upload).

## Pipeline overview

The pipeline runs in two stages:

**Stage 1 — raw extraction (`main.py`)**, run against the expedition root:

| Step | Module | Input | Output (per dive) |
|------|--------|-------|-------------------|
| 1 | `processors/dive_summaries.py` | `processed/dive_reports/<DIVE>/` stats + summary | `RUMI_processed/all_dive_summaries.csv` |
| 2 | `processors/process_dat.py` | `raw/nav/navest/*.DAT` (OCT + VFR lines) | `<EXP>_<DIVE>_pitch_roll_heading_octans.csv`, `<EXP>_<DIVE>_dvl_lat_long.csv` |
| 3 | `processors/usbl_sdyn.py` | `raw/datalog/*.SDYN` (GPGGA) | `<EXP>_<DIVE>_USBL_Hercules.csv` |
| 4 | `processors/sensors_sealog.py` | CTD/O2S/DEP sampled TSVs + sealog export | `<EXP>_<DIVE>_sealog_sensors_merged.csv`, `<EXP>_<DIVE>_USBL_Atalanta.csv` |
| 5 | `processors/stillcam_images.py` | `processed/capture_pngs/` | `stillcam_images/*.jpg` |

```bash
python main.py --dir Z:/NA173
```

**Stage 2 — Kalman pipeline (`main_kalman.py`)**, run per dive against
`<base>/<EXPEDITION>/RUMI_processed/<DIVE>`:

| Step | Module | Purpose |
|------|--------|---------|
| 1 | `processors/kalman_concat.py` | Outer-merge octans + USBL + DVL + sensors on Timestamp; 3σ pitch/roll outlier cull → `<EXP>_<DIVE>_filtered_datatable.csv` |
| 2 | `processors/kalman_filter.py` | 8-state Kalman filter (x, y, z, roll, pitch, vx, vy, vz) + circular heading smoother → `<EXP>_<DIVE>_kalman_filtered_data.csv`, `<EXP>_<DIVE>_final_datatable.csv` |
| 3 | `processors/kalman_assess.py` | Smoothness/consistency metrics + plots → `<EXP>_<DIVE>_kalman_assessment.csv` |
| 4 | `processors/kalman_offset.py` | Offset position 2 m backwards along heading, enforce ≥1 m terrain clearance against dive GeoTIFF → `<EXP>_<DIVE>_filtered_offset_final.csv` |

```bash
python main_kalman.py --base Z:/ --expedition NA173 --dive H2075 --yes
```

(Omit the flags to be prompted interactively.)

## Directory conventions

```
<base>/<EXPEDITION>/               # e.g. Z:/NA173
├── raw/
│   ├── nav/navest/*.DAT           # NavEst OCT + VFR records
│   ├── datalog/*.SDYN             # Sonardyne USBL GPGGA sentences
│   └── sealog/sealog-herc/<DIVE>/<DIVE>_sealogExport.csv
├── processed/
│   ├── dive_reports/<DIVE>/       # <DIVE>-stats.tsv, <DIVE>-summary.txt, sampled/
│   └── capture_pngs/capture_YYYYMMDD/
└── RUMI_processed/                # all pipeline output
    ├── all_dive_summaries.csv
    └── <DIVE>/                    # per-dive outputs + <DIVE>_k2mapping_geotiff_*.tif
```

## Data-handling conventions

All processors follow the rules in `processors/common.py`:

* **Timestamps** are UTC, ISO8601 `YYYY-MM-DDTHH:MM:SSZ`, no subseconds.
* **Second alignment** rounds to the *nearest* second (never truncates).
* When several fixes fall in one second, keep the best one
  (lowest USBL `Accuracy`, otherwise closest to the whole second).
* **Every CSV is written in chronological order** with unique timestamps.
* Depths are negative down (meters); headings are compass bearings
  (0° = North, clockwise); UTM x = easting, y = northing.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Known caveats

* The USBL "Accuracy" field is taken from the GPGGA slot that is HDOP in the
  standard sentence; the Kalman filter squares it and uses it as measurement
  variance in m². Verify against the Sonardyne SDYN output spec.
* `kalman_offset.py` expects the dive GeoTIFF's CRS to match the UTM zone the
  Kalman filter selected from the data (it prints both so you can check).
* If no USBL/DVL coordinates convert to UTM, `kalman_filter.py` falls back to
  UTM zone 4N.
