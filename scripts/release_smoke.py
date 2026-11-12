"""Run the release checks, including an isolated end-to-end sample pipeline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "reports/generated/checkpoints/20260905T1712Z-final"
CHECKPOINT = CHECKPOINT_ROOT / "checkpoint.json"
STATIC_CHECK_TARGETS = (
    "src",
    "tests",
    "scripts/generate_final_figures.py",
    "scripts/release_smoke.py",
)


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    _run("-m", "ruff", "format", "--check", *STATIC_CHECK_TARGETS)
    _run("-m", "ruff", "check", *STATIC_CHECK_TARGETS)
    _run("-m", "pytest")

    with tempfile.TemporaryDirectory(prefix="orderflow-mm-release-") as temporary:
        temporary_root = Path(temporary)
        raw_root = temporary_root / "raw"
        bars_root = temporary_root / "bars"
        validation_report = temporary_root / "validation.json"
        comparison_report = temporary_root / "comparison.json"
        _run(
            "-m",
            "orderflow_mm",
            "generate-sample",
            "--output",
            str(raw_root),
            "--seconds",
            "180",
        )
        _run("-m", "orderflow_mm", "qa", "--input", str(raw_root))
        _run(
            "-m",
            "orderflow_mm",
            "build-bars",
            "--input",
            str(raw_root),
            "--output",
            str(bars_root),
        )
        _run("-m", "orderflow_mm", "inspect-bars", "--input", str(bars_root))
        _run(
            "-m",
            "orderflow_mm",
            "validate-checkpoint",
            "--checkpoint",
            str(CHECKPOINT),
            "--report",
            str(validation_report),
        )
        _run(
            "-m",
            "orderflow_mm",
            "compare-simulations",
            "--baseline",
            str(CHECKPOINT_ROOT / "simulation-symmetric-queue-aware.json"),
            "--candidate",
            str(CHECKPOINT_ROOT / "simulation-signal_inventory-queue-aware.json"),
            "--bootstrap-samples",
            "2000",
            "--report",
            str(comparison_report),
        )

    print(
        "Release smoke test passed: static checks, tests, sample pipeline, final validation, "
        "and paired simulation comparison."
    )


if __name__ == "__main__":
    main()
