"""
Edge-case tests for the ROV data pipeline.

Run from the repository root:  python -m pytest tests/ -v
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processors.common import (
    best_fix_per_second,
    drop_duplicate_timestamps,
    determine_utm_zone,
)
from processors.usbl_sdyn import parse_sdyn_file
from processors.process_dat import parse_dat_file_both
from processors.dive_summaries import process_dive_folder


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# common.best_fix_per_second
# ---------------------------------------------------------------------------

class TestBestFixPerSecond:
    def test_rounds_to_nearest_second(self):
        df = pd.DataFrame({
            "Timestamp": [utc(2024, 11, 5, 12, 0, 0, 400000),
                          utc(2024, 11, 5, 12, 0, 1, 600000)],
            "v": [1.0, 2.0],
        })
        out, orig, final = best_fix_per_second(df)
        assert list(out["Timestamp"]) == ["2024-11-05T12:00:00Z", "2024-11-05T12:00:02Z"]
        assert (orig, final) == (2, 2)

    def test_closest_fix_wins_without_quality(self):
        df = pd.DataFrame({
            "Timestamp": [utc(2024, 11, 5, 12, 0, 0, 450000),   # 0.45s from second
                          utc(2024, 11, 5, 12, 0, 0, 100000)],  # 0.10s from second
            "v": [1.0, 2.0],
        })
        out, _, _ = best_fix_per_second(df)
        assert len(out) == 1
        assert out.iloc[0]["v"] == 2.0

    def test_best_quality_wins(self):
        df = pd.DataFrame({
            "Timestamp": [utc(2024, 11, 5, 12, 0, 0, 100000),
                          utc(2024, 11, 5, 12, 0, 0, 200000)],
            "Accuracy": [5.0, 1.0],
            "v": [1.0, 2.0],
        })
        out, _, _ = best_fix_per_second(df, quality_col="Accuracy")
        assert len(out) == 1
        assert out.iloc[0]["v"] == 2.0

    def test_all_nan_quality_group_does_not_crash(self):
        df = pd.DataFrame({
            "Timestamp": [utc(2024, 11, 5, 12, 0, 0), utc(2024, 11, 5, 12, 0, 0, 300000)],
            "Accuracy": [np.nan, np.nan],
            "v": [1.0, 2.0],
        })
        out, _, _ = best_fix_per_second(df, quality_col="Accuracy")
        assert len(out) == 1

    def test_rounding_collision_keeps_better_quality(self):
        # 11:59:59.6 and 12:00:00.4 both round to 12:00:00
        df = pd.DataFrame({
            "Timestamp": [utc(2024, 11, 5, 11, 59, 59, 600000),
                          utc(2024, 11, 5, 12, 0, 0, 400000)],
            "Accuracy": [9.0, 2.0],
            "v": [1.0, 2.0],
        })
        out, _, _ = best_fix_per_second(df, quality_col="Accuracy")
        assert len(out) == 1
        assert out.iloc[0]["v"] == 2.0

    def test_output_is_chronological_and_unique(self):
        rng = np.random.default_rng(0)
        base = pd.Timestamp("2024-11-05T12:00:00Z")
        ts = [base + pd.Timedelta(seconds=float(s)) for s in rng.uniform(0, 100, 300)]
        df = pd.DataFrame({"Timestamp": ts, "Accuracy": rng.uniform(1, 20, 300)})
        out, _, _ = best_fix_per_second(df, quality_col="Accuracy")
        assert out["Timestamp"].is_monotonic_increasing
        assert not out["Timestamp"].duplicated().any()

    def test_object_dtype_timestamps_accepted(self):
        df = pd.DataFrame({
            "Timestamp": pd.Series([utc(2024, 11, 5, 12, 0, 0, 300000), "garbage"],
                                   dtype=object),
            "v": [1.0, 2.0],
        })
        out, orig, final = best_fix_per_second(df)
        assert orig == 2 and final == 1
        assert out.iloc[0]["v"] == 1.0

    def test_empty_frame(self):
        df = pd.DataFrame({"Timestamp": pd.Series([], dtype="datetime64[ns, UTC]")})
        out, orig, final = best_fix_per_second(df)
        assert out.empty and orig == 0 and final == 0


class TestDropDuplicateTimestamps:
    def test_sorted_and_deduped(self):
        df = pd.DataFrame({
            "Timestamp": ["2024-11-05T12:00:02Z", "2024-11-05T12:00:00Z",
                          "2024-11-05T12:00:02Z"],
            "v": [1, 2, 3],
        })
        out, removed = drop_duplicate_timestamps(df)
        assert removed == 1
        assert list(out["Timestamp"]) == ["2024-11-05T12:00:00Z", "2024-11-05T12:00:02Z"]

    def test_none_and_empty(self):
        assert drop_duplicate_timestamps(None) == (None, 0)
        empty = pd.DataFrame({"Timestamp": []})
        out, removed = drop_duplicate_timestamps(empty)
        assert out.empty and removed == 0


class TestUtmZone:
    def test_regular_zones(self):
        assert determine_utm_zone(133.95, 5.67) == (53, "north")   # Palau (NA167)
        assert determine_utm_zone(-157.0, 20.0) == (4, "north")    # Hawaii
        assert determine_utm_zone(-70.0, -30.0) == (19, "south")

    def test_norway_svalbard_exceptions(self):
        assert determine_utm_zone(5.0, 60.0)[0] == 32
        assert determine_utm_zone(10.0, 75.0)[0] == 33


# ---------------------------------------------------------------------------
# usbl_sdyn.parse_sdyn_file
# ---------------------------------------------------------------------------

def gpgga(time_s, lat="0540.350", ns="N", lon="13356.905", ew="E",
          acc="14.5", depth="-1020.5", beacon="0001"):
    return (f"$GPGGA,{time_s},{lat},{ns},{lon},{ew},1,08,{acc},{depth},"
            f"M,0.0,M,0.0,{beacon}*5A")


class TestParseSdyn:
    def test_basic_parse_and_decimal_degrees(self, tmp_path):
        f = tmp_path / "20241105_2310.SDYN"
        f.write_text(gpgga("231205.50") + "\n")
        df = parse_sdyn_file(f)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["Vehicle"] == "Hercules"
        assert row["Timestamp"] == pd.Timestamp("2024-11-05T23:12:05.5Z")
        assert row["Latitude"] == pytest.approx(5 + 40.350 / 60)
        assert row["Longitude"] == pytest.approx(133 + 56.905 / 60)
        assert row["Accuracy"] == pytest.approx(14.5)

    def test_south_west_are_negative(self, tmp_path):
        f = tmp_path / "20241105_2310.SDYN"
        f.write_text(gpgga("231205.50", ns="S", ew="W") + "\n")
        row = parse_sdyn_file(f).iloc[0]
        assert row["Latitude"] < 0 and row["Longitude"] < 0

    def test_midnight_rollover(self, tmp_path):
        f = tmp_path / "20241105_2350.SDYN"
        f.write_text(gpgga("235959.00") + "\n" + gpgga("000010.00") + "\n")
        df = parse_sdyn_file(f)
        assert df.iloc[0]["Timestamp"] == pd.Timestamp("2024-11-05T23:59:59Z")
        assert df.iloc[1]["Timestamp"] == pd.Timestamp("2024-11-06T00:00:10Z")

    def test_atalanta_beacon_skipped(self, tmp_path):
        f = tmp_path / "20241105_2310.SDYN"
        f.write_text(gpgga("231205.50", beacon="0002") + "\n"
                     + gpgga("231206.50", beacon="0001") + "\n")
        df = parse_sdyn_file(f)
        assert len(df) == 1
        assert df.iloc[0]["Vehicle"] == "Hercules"

    def test_short_time_field_zero_padded(self, tmp_path):
        # 12345.60 must parse as 01:23:45.6, not 12:34:5.6
        f = tmp_path / "20241105_0100.SDYN"
        f.write_text(gpgga("12345.60") + "\n")
        df = parse_sdyn_file(f)
        assert df.iloc[0]["Timestamp"] == pd.Timestamp("2024-11-05T01:23:45.6Z")

    def test_bad_filename_returns_empty(self, tmp_path):
        f = tmp_path / "notadate.SDYN"
        f.write_text(gpgga("231205.50") + "\n")
        assert parse_sdyn_file(f).empty


# ---------------------------------------------------------------------------
# process_dat.parse_dat_file_both
# ---------------------------------------------------------------------------

def oct_line(ts="2024/11/05 23:13:02.123", heading=228.42, pitch=-6.5, roll=-1.05):
    skip3 = "1.0 2.0 3.0"
    return (f"OCT {ts} Hercules {skip3} {heading} {pitch} {roll} "
            f"{skip3} 4.0 5.0 6.0 7.0 8.0 9.0")


def vfr_line(ts="2024/11/05 23:13:02.50", vehicle="0", fix="SOLN_DEADRECK",
             lon=133.948, lat=5.672):
    return f"VFR {ts} 123 {vehicle} {fix} {lon} {lat}"


class TestParseDat:
    def test_oct_and_vfr_extracted(self, tmp_path):
        f = tmp_path / "test.DAT"
        f.write_text(oct_line() + "\n" + vfr_line() + "\n")
        oct_df, vfr_df = parse_dat_file_both(f)
        assert len(oct_df) == 1 and len(vfr_df) == 1
        assert oct_df.iloc[0]["Heading"] == pytest.approx(228.42)
        assert oct_df.iloc[0]["Pitch"] == pytest.approx(-6.5)
        assert oct_df.iloc[0]["Roll"] == pytest.approx(-1.05)

    def test_wrong_vehicle_or_fix_type_skipped(self, tmp_path):
        f = tmp_path / "test.DAT"
        f.write_text(vfr_line(vehicle="1") + "\n"
                     + vfr_line(fix="SOLN_USBL") + "\n")
        _, vfr_df = parse_dat_file_both(f)
        assert vfr_df.empty

    def test_null_island_fix_rejected(self, tmp_path):
        f = tmp_path / "test.DAT"
        f.write_text(vfr_line(lon=0.001, lat=-0.002) + "\n")
        _, vfr_df = parse_dat_file_both(f)
        assert vfr_df.empty

    def test_malformed_second_field_skipped(self, tmp_path):
        f = tmp_path / "test.DAT"
        f.write_text(oct_line(ts="2024/11/05 23:13:61.0") + "\n" + oct_line() + "\n")
        oct_df, _ = parse_dat_file_both(f)
        assert len(oct_df) == 1


# ---------------------------------------------------------------------------
# dive_summaries.process_dive_folder
# ---------------------------------------------------------------------------

STATS_HEADER = ("##NA167\tdive\tsite\tinwatertime\tonbottomtime\toffbottomtime"
                "\tondecktime\thercmaxdepth\thercavgdepth\ttotaltime(hours)"
                "\tbottomtime(hours)")


def write_dive(tmp_path, dive="H2075", total_hours="26.4"):
    folder = tmp_path / dive
    folder.mkdir()
    row = (f"NA167\t{dive}\tSome_Site\t2024-11-05T23:13:02Z\t2024-11-06T00:08:00Z"
           f"\t2024-11-07T00:10:00Z\t2024-11-07T01:37:00Z\t-1327.1\t-1005.0"
           f"\t{total_hours}\t20.0")
    (folder / f"{dive}-stats.tsv").write_text(STATS_HEADER + "\n" + row + "\n")
    (folder / f"{dive}-summary.txt").write_text("Objective: Map the site.\n")
    return folder


class TestDiveSummaries:
    def test_no_duplicate_columns(self, tmp_path):
        folder = write_dive(tmp_path)
        df = process_dive_folder(folder, "H2075")
        assert df is not None
        assert not df.columns.duplicated().any()
        assert df.iloc[0]["Objective"] == "Map the site."
        # Recovery Time is the recorded on-deck time, Dive End is derived.
        assert df.iloc[0]["Recovery Time"] == "2024-11-07T01:37:00Z"
        assert df.iloc[0]["Dive End"] == "2024-11-07T01:37:02Z"  # launch + 26.4 h
        assert df.iloc[0]["site"] == "Some Site"

    def test_short_dive_skipped(self, tmp_path):
        folder = write_dive(tmp_path, total_hours="1.5")
        assert process_dive_folder(folder, "H2075") is None

    def test_missing_files_skipped(self, tmp_path):
        folder = tmp_path / "H9999"
        folder.mkdir()
        assert process_dive_folder(folder, "H9999") is None


# ---------------------------------------------------------------------------
# Resume / restart support
# ---------------------------------------------------------------------------

class TestResume:
    def test_kalman_module_output_paths(self):
        from main_kalman import module_output_path
        processed = Path("Z:/NA167/RUMI_processed/H2075")
        p = module_output_path("kalman_concat", processed)
        assert p.name == "NA167_H2075_filtered_datatable.csv"
        p = module_output_path("kalman_offset", processed)
        assert p.name == "NA167_H2075_filtered_offset_final.csv"
        assert module_output_path("unknown_module", processed) is None

    def test_stage1_output_detection(self, tmp_path):
        from main import step_outputs_exist
        root = tmp_path
        (root / "RUMI_processed" / "H2075").mkdir(parents=True)
        assert not step_outputs_exist("process_dat", root)
        (root / "RUMI_processed" / "H2075" /
         "NA167_H2075_pitch_roll_heading_octans.csv").write_text("Timestamp\n")
        assert step_outputs_exist("process_dat", root)
        assert not step_outputs_exist("dive_summaries", root)
        (root / "RUMI_processed" / "all_dive_summaries.csv").write_text("x\n")
        assert step_outputs_exist("dive_summaries", root)
        # stillcam has per-image resume, never step-level skip
        assert not step_outputs_exist("stillcam_images", root)
