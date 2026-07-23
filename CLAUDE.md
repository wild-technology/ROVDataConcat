# ROVDataConcat — context for Claude sessions

Two-stage pipeline for Hercules/Atalanta ROV expedition data (OET/Nautilus
style, e.g. NA167, NA173). Output feeds Unreal Engine dive visualizations.
See README.md for the full pipeline table, directory layout, and run commands.

## Quick facts

- Stage 1: `python main.py --dir <expedition_root>` (raw extraction).
- Stage 2: `python main_kalman.py --base <base> --expedition <EXP> --dive <DIVE> --yes`.
- Both orchestrators auto-skip steps whose outputs exist (`--force` to redo).
- Tests: `python -m pytest tests/` (must stay green; run before pushing).
- Every stage prints a Data Quality Report and writes a JSON provenance
  sidecar under `RUMI_processed/[<DIVE>/]reports/`.

## Non-negotiable data conventions (enforced via processors/common.py)

- Timestamps: UTC, ISO8601 `YYYY-MM-DDTHH:MM:SSZ`, no subseconds.
- Second alignment rounds to *nearest* second; never truncate.
- Every output CSV is chronological with unique timestamps.
- Depths are negative down (meters). Heading is a compass bearing
  (0° = North, clockwise) — VERIFIED empirically against DVL course-over-
  ground on NA167/H2075; never treat it as a math angle. In UTM,
  forward unit vector = (sin h, cos h).
- USBL "Accuracy" (GPGGA HDOP slot) is a positional accuracy in meters
  (~1.4% of slant range empirically), used as sigma for Kalman R.

## History / decisions (July 2026 overhaul)

- DVL gate is `depth <= -30` (bottom lock). It was inverted (`>= -30`)
  before, which silently discarded ALL DVL data.
- The 2 m backwards vehicle offset (kalman_offset.py) previously used
  math-angle trig and pointed the wrong way; outputs produced before commit
  638ac58 have offset positions in the wrong direction and pre-2bf91dc
  tracks contain no DVL influence — regenerate before trusting.
- Pitch/roll 3σ outliers are NULLED, not row-dropped (dropping rows lost
  sealog annotations).
- kalman_* outputs are RTS-smoothed (forward-backward), not causal-filtered.
- GeoTIFF is found by glob `<DIVE>_k2mapping_geotiff*.tif` and sampled via
  a CRS transform; off-raster/nodata samples impose no depth constraint.
- requirements.txt must stay UTF-8/ASCII (it was once UTF-16, which pip
  cannot read; beware editors that preserve existing file encoding).

## Working agreements

- Owner: Jonathan (jonathan@wildtechnology.org). Commit directly to main —
  no PRs on this repo.
- Console output must stay ASCII (Windows cp1252 consoles crash on arrows,
  bullets, and emoji in prints).
- Example production output data for validation: a per-dive folder like
  D:\na167_h2075 (dive H2075, expedition NA167, Palau, ~5.67°N 133.95°E,
  UTM zone 53N).
