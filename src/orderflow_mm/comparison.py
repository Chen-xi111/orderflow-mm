from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def compare_simulation_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two simulation scenarios using paired UTC-hour bootstrap resampling.

    Each reported UTC hour is treated as one dependence block. Hours present in only one
    scenario are retained with zero fill economics for the other scenario. Hours with no fills
    in either scenario cannot be reconstructed from summary reports and are outside the estimand.
    """
    if bootstrap_samples < 1_000:
        raise ValueError("bootstrap_samples must be at least 1000")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1.0")

    baseline_config = _validated_config(baseline, "baseline")
    candidate_config = _validated_config(candidate, "candidate")
    baseline_controls = {key: value for key, value in baseline_config.items() if key != "strategy"}
    candidate_controls = {
        key: value for key, value in candidate_config.items() if key != "strategy"
    }
    if baseline_controls != candidate_controls:
        raise ValueError("simulation controls must match except for strategy")
    if baseline_config["strategy"] == candidate_config["strategy"]:
        raise ValueError("baseline and candidate strategies must differ")
    if baseline.get("start") != candidate.get("start") or baseline.get("end") != candidate.get(
        "end"
    ):
        raise ValueError("simulation start and end timestamps must match")

    baseline_hourly = _hourly_net_fill_economics(baseline, "baseline")
    candidate_hourly = _hourly_net_fill_economics(candidate, "candidate")
    hours = sorted(set(baseline_hourly) | set(candidate_hourly))
    if len(hours) < 2:
        raise ValueError("comparison requires at least two active UTC hours")

    baseline_values = np.asarray([baseline_hourly.get(hour, 0.0) for hour in hours])
    candidate_values = np.asarray([candidate_hourly.get(hour, 0.0) for hour in hours])
    differences = candidate_values - baseline_values

    rng = np.random.default_rng(seed)
    resampled_indices = rng.integers(0, len(hours), size=(bootstrap_samples, len(hours)))
    bootstrap_means = differences[resampled_indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    probability_better = float(np.mean(bootstrap_means > 0.0))

    if lower > 0.0:
        decision = "candidate_supported"
    elif upper < 0.0:
        decision = "candidate_harmed"
    else:
        decision = "inconclusive"

    hourly_rows = [
        {
            "hour_utc": hour,
            "baseline_net_fill_economics": float(baseline_values[index]),
            "candidate_net_fill_economics": float(candidate_values[index]),
            "difference": float(differences[index]),
        }
        for index, hour in enumerate(hours)
    ]
    return {
        "schema_version": 1,
        "method": {
            "name": "paired_nonparametric_utc_hour_bootstrap",
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "seed": seed,
            "estimand": "mean hourly net one-second fill markout after maker fees",
            "hour_population": "UTC hours with fills in at least one compared scenario",
        },
        "comparison": {
            "baseline_strategy": baseline_config["strategy"],
            "candidate_strategy": candidate_config["strategy"],
            "fill_model": baseline_config["fill_model"],
            "maker_fee_bps": float(baseline_config["maker_fee_bps"]),
            "start": baseline["start"],
            "end": baseline["end"],
        },
        "hours": len(hours),
        "hours_candidate_better": int(np.sum(differences > 0.0)),
        "hours_candidate_worse": int(np.sum(differences < 0.0)),
        "hours_tied": int(np.sum(differences == 0.0)),
        "baseline": _scenario_summary(baseline_values),
        "candidate": _scenario_summary(candidate_values),
        "difference": {
            "total_net_fill_economics": float(differences.sum()),
            "mean_hourly_net_fill_economics": float(differences.mean()),
            "confidence_interval": [float(lower), float(upper)],
            "bootstrap_probability_candidate_better": probability_better,
        },
        "decision": decision,
        "automatic_profitability_claims_are_disabled": True,
        "limitations": [
            "The interval describes scenario differences, not executable trading returns.",
            "Top-of-book data cannot reconstruct price-level queue position.",
            (
                "Net fill economics exclude inventory mark-to-market, gap liquidation, "
                "and final liquidation."
            ),
            "Hours with no fills in either scenario are absent from summary-report attribution.",
            "Resampling UTC-hour blocks does not remove regime-selection or model-selection risk.",
        ],
        "hourly_differences": hourly_rows,
    }


def load_simulation_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"simulation report must contain a JSON object: {path}")
    return report


def write_comparison_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _validated_config(report: dict[str, Any], name: str) -> dict[str, Any]:
    config = report.get("config")
    required = {"strategy", "fill_model", "maker_fee_bps"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"{name} report is missing simulation config fields")
    if not isinstance(report.get("hourly_fill_attribution"), list):
        raise ValueError(f"{name} report is missing hourly fill attribution")
    if not isinstance(report.get("start"), str) or not isinstance(report.get("end"), str):
        raise ValueError(f"{name} report is missing simulation timestamps")
    return config


def _hourly_net_fill_economics(report: dict[str, Any], name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in report["hourly_fill_attribution"]:
        try:
            hour = str(row["hour_utc"])
            value = float(row["markout_1s"]) - float(row["total_fees"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} report has invalid hourly attribution") from exc
        if hour in result:
            raise ValueError(f"{name} report contains duplicate hour: {hour}")
        if not np.isfinite(value):
            raise ValueError(f"{name} report contains non-finite hourly economics")
        result[hour] = value
    if "markout_1s" in report and "total_fees" in report:
        expected_total = float(report["markout_1s"]) - float(report["total_fees"])
        if not np.isclose(sum(result.values()), expected_total, rtol=1e-10, atol=1e-8):
            raise ValueError(f"{name} hourly attribution does not reconcile to aggregate totals")
    return result


def _scenario_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "total_net_fill_economics": float(values.sum()),
        "mean_hourly_net_fill_economics": float(values.mean()),
        "median_hourly_net_fill_economics": float(np.median(values)),
    }
