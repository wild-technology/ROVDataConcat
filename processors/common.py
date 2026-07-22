"""
Shared helpers for the ROV data pipeline.

Centralizes behavior that was previously duplicated (with drift) across
processor modules:

* ISO8601 timestamp formatting ("YYYY-MM-DDTHH:MM:SSZ", UTC, no subseconds)
* Second-alignment of high-rate fixes (round to *nearest* second everywhere)
* Duplicate-timestamp removal (always returns chronologically sorted data)
* Deriving <expedition>/<dive> identifiers from the processed directory
"""

from pathlib import Path

import pandas as pd

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def to_iso8601(dt_series: pd.Series) -> pd.Series:
    """Format a datetime Series as ISO8601 UTC strings without subseconds."""
    return dt_series.apply(
        lambda dt: dt.strftime(ISO_FMT) if pd.notnull(dt) else dt
    )


def parse_utc(series: pd.Series) -> pd.Series:
    """Parse a Series to timezone-aware UTC datetimes, coercing failures to NaT."""
    return pd.to_datetime(series.astype(str).str.strip(), utc=True, errors="coerce")


def drop_duplicate_timestamps(df: pd.DataFrame, sort_by: str = "Timestamp"):
    """
    Drop rows with duplicate timestamps (keep first) and return the frame
    sorted chronologically.

    Returns (df, removed_count).
    """
    if df is None or df.empty:
        return df, 0
    before = len(df)
    out = df.drop_duplicates(subset=["Timestamp"]).sort_values(sort_by, kind="mergesort")
    return out, before - len(out)


def best_fix_per_second(df: pd.DataFrame, quality_col: str = None):
    """
    Align fixes to whole seconds by rounding Timestamp to the nearest second,
    then keep one row per second:

    * quality_col given  -> row with the lowest value in that column
      (e.g. USBL 'Accuracy'),
    * otherwise          -> the fix whose original time is closest to the
      rounded second.

    The Timestamp column of the result is an ISO8601 string. The result is
    sorted chronologically. Returns (df, original_count, final_count).
    """
    if df.empty:
        return df.copy(), 0, 0

    orig = len(df)
    work = df.copy()
    rounded = work["Timestamp"].dt.round("s")

    if quality_col is not None:
        work["_rounded"] = rounded
        keep_idx = work.groupby("_rounded")[quality_col].idxmin()
    else:
        work["_rounded"] = rounded
        work["_diff"] = (work["Timestamp"] - rounded).abs()
        keep_idx = work.groupby("_rounded")["_diff"].idxmin()

    out = work.loc[keep_idx].copy()
    out.sort_values("_rounded", inplace=True)
    out["Timestamp"] = out["_rounded"].dt.strftime(ISO_FMT)
    out.drop(columns=[c for c in ("_rounded", "_diff") if c in out.columns], inplace=True)
    # Rounding can map two source seconds onto one target second; keep first.
    out = out.drop_duplicates(subset=["Timestamp"])
    out.reset_index(drop=True, inplace=True)
    return out, orig, len(out)


def determine_utm_zone(lon, lat):
    """Determine the UTM zone (number, hemisphere) for a lon/lat coordinate."""
    zone_number = int((lon + 180) / 6) + 1

    # Special cases for Norway and Svalbard
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone_number = 32
    if 72.0 <= lat < 84.0:
        if 0.0 <= lon < 9.0:
            zone_number = 31
        elif 9.0 <= lon < 21.0:
            zone_number = 33
        elif 21.0 <= lon < 33.0:
            zone_number = 35
        elif 33.0 <= lon < 42.0:
            zone_number = 37

    hemisphere = "north" if lat >= 0 else "south"
    return zone_number, hemisphere


def utm_proj_string(lon, lat):
    """proj4 string for the WGS84 UTM zone containing lon/lat."""
    zone_number, hemisphere = determine_utm_zone(lon, lat)
    return f"+proj=utm +zone={zone_number} +{hemisphere} +datum=WGS84 +units=m +no_defs"


def expedition_dive_from_processed_dir(processed_dir: Path):
    """
    Derive (expedition, dive) from the standardized layout
    <base>/<EXPEDITION>/RUMI_processed/<DIVE>.
    """
    processed_dir = Path(processed_dir).resolve()
    dive = processed_dir.name
    expedition = processed_dir.parent.parent.name
    return expedition, dive
