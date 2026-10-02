import json
from pathlib import Path

import pytest

from orderflow_mm.comparison import compare_simulation_reports, write_comparison_report


def _report(strategy: str, hourly_values: list[float], fill_model: str = "queue_aware") -> dict:
    return {
        "config": {
            "strategy": strategy,
            "fill_model": fill_model,
            "maker_fee_bps": 1.0,
            "order_size": 0.001,
        },
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-01-01T04:00:00+00:00",
        "hourly_fill_attribution": [
            {
                "hour_utc": f"2026-01-01T{index:02d}:00:00+00:00",
                "markout_1s": value + 1.0,
                "total_fees": 1.0,
            }
            for index, value in enumerate(hourly_values)
        ],
    }


def test_hourly_bootstrap_supports_uniform_candidate_improvement() -> None:
    baseline = _report("symmetric", [-2.0, -1.0, 0.0, 1.0])
    candidate = _report("signal_inventory", [-1.0, 0.0, 1.0, 2.0])

    report = compare_simulation_reports(baseline, candidate, bootstrap_samples=2_000, seed=7)

    assert report["decision"] == "candidate_supported"
    assert report["difference"]["total_net_fill_economics"] == 4.0
    assert report["difference"]["confidence_interval"] == [1.0, 1.0]
    assert report["automatic_profitability_claims_are_disabled"] is True


def test_hourly_bootstrap_reports_mixed_difference_as_inconclusive() -> None:
    baseline = _report("symmetric", [0.0, 0.0, 0.0, 0.0])
    candidate = _report("signal_inventory", [-2.0, -1.0, 1.0, 2.0])

    first = compare_simulation_reports(baseline, candidate, bootstrap_samples=2_000, seed=11)
    second = compare_simulation_reports(baseline, candidate, bootstrap_samples=2_000, seed=11)

    assert first["decision"] == "inconclusive"
    assert first["difference"] == second["difference"]
    assert first["hours_candidate_better"] == 2
    assert first["hours_candidate_worse"] == 2


def test_comparison_rejects_confounded_controls() -> None:
    baseline = _report("symmetric", [0.0, 1.0])
    candidate = _report("signal_inventory", [1.0, 2.0])
    candidate["config"]["maker_fee_bps"] = 0.0

    with pytest.raises(ValueError, match="controls must match"):
        compare_simulation_reports(baseline, candidate)


def test_comparison_rejects_unreconciled_hourly_attribution() -> None:
    baseline = _report("symmetric", [0.0, 1.0])
    candidate = _report("signal_inventory", [1.0, 2.0])
    baseline["markout_1s"] = 0.0
    baseline["total_fees"] = 0.0

    with pytest.raises(ValueError, match="does not reconcile"):
        compare_simulation_reports(baseline, candidate)


def test_comparison_writer_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "comparison.json"
    write_comparison_report({"decision": "inconclusive"}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"decision": "inconclusive"}
    assert not path.with_suffix(".json.tmp").exists()
