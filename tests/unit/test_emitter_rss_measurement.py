from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.check_emitter_rss import (
    GateError,
    evaluate_runs,
    load_json,
    selected_cases,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/perf/emitter/cases.json"
LIMITS = ROOT / "tests/perf/emitter/rss_limits.json"


def _run(peak_kib: int = 64 * 1024, **overrides) -> dict:
    result = {
        "exit_status": 0,
        "timed_out": False,
        "rss": {
            "available": True,
            "peak_sampled_sum_kib": peak_kib,
            "unavailable_reason": None,
        },
    }
    result.update(overrides)
    return result


def test_checked_in_cases_and_limits_cover_both_tiers_and_targets() -> None:
    cases = load_json(CASES)
    limits = load_json(LIMITS)

    gate_names = {name for name, _case in selected_cases(cases, "gate")}
    nightly_names = {name for name, _case in selected_cases(cases, "nightly")}
    assert gate_names
    assert gate_names < nightly_names

    linux = limits["linux-x86_64"]["release"]
    for name, case in cases["cases"].items():
        assert case["axis"] in {"shared", "distinct"}
        assert case["size"] > 0
        assert 0 < case["final_ac_min_bytes"] < case["final_ac_max_bytes"]
        for target in ("cpp", "verilog"):
            value = linux[name][f"{target}_mib"]
            assert isinstance(value, int) and not isinstance(value, bool) and value > 0


def test_evaluate_runs_reports_median_and_gates_on_maximum() -> None:
    result = evaluate_runs([_run(60 * 1024), _run(80 * 1024), _run(64 * 1024)], 72)

    assert result["status"] == "fail"
    assert result["median_peak_rss_kib"] == 64 * 1024
    assert result["max_peak_rss_kib"] == 80 * 1024
    assert "exceeds 72 MiB" in result["failures"][-1]


@pytest.mark.parametrize(
    "run, message",
    [
        (_run(timed_out=True), "timed out"),
        (_run(exit_status=2), "exited 2"),
        (
            _run(
                rss={
                    "available": False,
                    "peak_sampled_sum_kib": None,
                    "unavailable_reason": "no samples",
                }
            ),
            "RSS unavailable: no samples",
        ),
    ],
)
def test_evaluate_runs_fails_invalid_measurements(run: dict, message: str) -> None:
    result = evaluate_runs([run], 128)

    assert result["status"] == "fail"
    assert any(message in failure for failure in result["failures"])


def test_load_json_rejects_an_unversioned_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"cases": {}}), encoding="utf-8")

    with pytest.raises(GateError, match="schema_version"):
        load_json(config)


def test_evaluate_runs_rejects_a_non_integer_limit() -> None:
    with pytest.raises(GateError, match="positive integer"):
        evaluate_runs([_run()], 128.5)  # type: ignore[arg-type]
