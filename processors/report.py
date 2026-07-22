"""
Run reporting: surfaces data anomalies to the console at the end of every
pipeline stage and writes a JSON provenance sidecar next to the outputs.

Usage in a processor:

    from processors.report import RunReport

    report = RunReport("kalman_concat", processed_dir)
    report.add_input(usbl_file, rows=len(usbl_df))
    report.metric("rows_merged", len(merged))
    report.anomaly("time-gaps", "3 gaps > 60s in merged timeline (max 542s)")
    ...
    report.add_output(output_file, rows=len(merged))
    report.finalize()          # prints the summary block, writes reports/<stage>.json
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_commit():
    """Best-effort git commit hash of the pipeline code (provenance)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


class RunReport:
    """Collects metrics/anomalies for one stage run; prints and persists them."""

    SEVERITIES = ("info", "warning", "anomaly", "error")

    def __init__(self, stage, report_dir=None):
        self.stage = stage
        self.report_dir = Path(report_dir) if report_dir else None
        self.started = datetime.now(timezone.utc)
        self.inputs = []
        self.outputs = []
        self.metrics = {}
        self.events = []  # {severity, category, message}

    # -- recording ----------------------------------------------------------
    def add_input(self, path, rows=None):
        self.inputs.append({"path": str(path), "rows": rows})

    def add_output(self, path, rows=None):
        self.outputs.append({"path": str(path), "rows": rows})

    def metric(self, name, value):
        self.metrics[name] = value

    def event(self, severity, category, message):
        assert severity in self.SEVERITIES
        self.events.append(
            {"severity": severity, "category": category, "message": str(message)}
        )

    def info(self, category, message):
        self.event("info", category, message)

    def warn(self, category, message):
        self.event("warning", category, message)

    def anomaly(self, category, message):
        self.event("anomaly", category, message)

    def error(self, category, message):
        self.event("error", category, message)

    # -- output -------------------------------------------------------------
    def _flagged(self):
        return [e for e in self.events if e["severity"] != "info"]

    def finalize(self):
        """Print the end-of-stage summary and write the JSON sidecar."""
        flagged = self._flagged()
        print(f"\n=== Data Quality Report: {self.stage} ===")
        if flagged:
            for e in flagged:
                tag = e["severity"].upper()
                print(f"  [{tag}] ({e['category']}) {e['message']}")
        else:
            print("  No anomalies detected.")
        for name, value in self.metrics.items():
            print(f"  - {name}: {value}")

        json_path = None
        if self.report_dir is not None:
            reports = self.report_dir / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            json_path = reports / f"{self.stage}.json"
            payload = {
                "stage": self.stage,
                "pipeline_commit": _git_commit(),
                "started_utc": self.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "inputs": self.inputs,
                "outputs": self.outputs,
                "metrics": self.metrics,
                "events": self.events,
            }
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"  Report saved: {json_path}")
        print("=" * (35 + len(self.stage)))
        return len(flagged)
