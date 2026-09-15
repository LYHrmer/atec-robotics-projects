"""CPU audit for :mod:`task_b.brake_wheel_hold`.

Run with::

    python -m task_b.audit_brake_wheel_hold --output results/task_b_brake_wheel_hold_cpu_audit.json

Exits non-zero if any check fails.  No simulator, GPU, or network access.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Callable, Dict, List

from task_b.brake_wheel_hold import BrakeWheelHold

TWO_PI = 2.0 * math.pi
CHECKS: List[Dict[str, Any]] = []


def check(name: str, fn: Callable[[], Any]) -> None:
    try:
        detail = fn()
        CHECKS.append({"name": name, "passed": True, "detail": detail})
    except Exception as exc:  # noqa: BLE001 - audit records every failure
        CHECKS.append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})


def rejects(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def settle(hold: BrakeWheelHold, q, qdot, steps: int) -> float:
    cmd = 0.0
    for _ in range(steps):
        cmd = hold.update(q, qdot)
    return cmd


def sign_positive() -> Dict[str, Any]:
    hold = BrakeWheelHold()
    hold.engage([1.0, 1.0, 1.0, 1.0])
    cmd = hold.update([0.9, 0.9, 0.9, 0.9], [0.0] * 4)
    assert cmd > 0.0, cmd
    assert all(e > 0.0 for e in hold.debug["wrapped_errors"])
    return {"command_rad_s": cmd}


def sign_negative() -> Dict[str, Any]:
    hold = BrakeWheelHold()
    hold.engage([1.0, 1.0, 1.0, 1.0])
    cmd = hold.update([1.1, 1.1, 1.1, 1.1], [0.0] * 4)
    assert cmd < 0.0, cmd
    assert all(e < 0.0 for e in hold.debug["wrapped_errors"])
    return {"command_rad_s": cmd}


def wrap_equivalence() -> Dict[str, Any]:
    base_q = [0.9, 0.8, 1.05, 0.95]
    qdot = [0.1, -0.1, 0.05, 0.0]
    results = {}
    for label, shift in (("+0", 0.0), ("+2pi", TWO_PI), ("-2pi", -TWO_PI), ("+4pi", 2 * TWO_PI), ("-4pi", -2 * TWO_PI)):
        hold = BrakeWheelHold()
        hold.engage([1.0] * 4)
        results[label] = settle(hold, [q + shift for q in base_q], qdot, 5)
    reference = results["+0"]
    for label, value in results.items():
        assert abs(value - reference) < 1e-9, (label, value, reference)
    return results


def speed_cap() -> Dict[str, Any]:
    hold = BrakeWheelHold()
    hold.engage([1.5] * 4)
    peak = 0.0
    for _ in range(200):
        peak = max(peak, abs(hold.update([0.0] * 4, [0.0] * 4)))
    assert peak <= hold.max_speed + 1e-12, peak
    assert abs(peak - hold.max_speed) < 1e-9, peak
    assert hold.debug["saturated"] is True
    return {"peak_rad_s": peak, "max_speed": hold.max_speed}


def slew_per_step() -> Dict[str, Any]:
    hold = BrakeWheelHold(dt=0.02, slew=2.0)
    hold.engage([1.5] * 4)
    limit = hold.slew * hold.dt
    previous = 0.0
    steps = []
    for _ in range(6):
        cmd = hold.update([0.0] * 4, [0.0] * 4)
        steps.append(cmd - previous)
        assert abs(cmd - previous) <= limit + 1e-12, (cmd, previous, limit)
        previous = cmd
    assert abs(steps[0] - limit) < 1e-12, steps[0]
    return {"slew_limit_rad_s": limit, "first_steps": steps}


def bad_inputs() -> Dict[str, Any]:
    hold = BrakeWheelHold()
    hold.engage([0.0] * 4)
    rejects(lambda: hold.update([float("nan")] * 4, [0.0] * 4))
    rejects(lambda: hold.update([0.0] * 4, [float("inf"), 0.0, 0.0, 0.0]))
    rejects(lambda: hold.update([0.0, 0.0, 0.0], [0.0] * 4))
    rejects(lambda: hold.update([0.0] * 5, [0.0] * 4))
    rejects(lambda: hold.update([[0.0], [0.0], [0.0], [0.0]], [0.0] * 4))
    rejects(lambda: hold.update([[0.0, 0.0], [0.0, 0.0]], [0.0] * 4))
    rejects(lambda: BrakeWheelHold().engage([0.0, 0.0, float("nan"), 0.0]))
    return {"rejected": 7}


def bad_parameters() -> Dict[str, Any]:
    rejects(lambda: BrakeWheelHold(dt=0.0))
    rejects(lambda: BrakeWheelHold(dt=float("inf")))
    rejects(lambda: BrakeWheelHold(kp=-1.0))
    rejects(lambda: BrakeWheelHold(kd=-0.1))
    rejects(lambda: BrakeWheelHold(kd=float("nan")))
    rejects(lambda: BrakeWheelHold(max_speed=0.0))
    rejects(lambda: BrakeWheelHold(slew=-2.0))
    BrakeWheelHold(kd=0.0)
    return {"rejected": 7, "kd_zero_allowed": True}


def engage_copies_input() -> Dict[str, Any]:
    hold = BrakeWheelHold()
    mutable = [1.0, 1.0, 1.0, 1.0]
    hold.engage(mutable)
    mutable[0] = 99.0
    assert hold.anchor == [1.0, 1.0, 1.0, 1.0], hold.anchor
    return {"anchor": hold.anchor}


def repeat_engage_does_not_relatch() -> Dict[str, Any]:
    hold = BrakeWheelHold()
    hold.engage([1.0] * 4)
    first = hold.update([0.9] * 4, [0.0] * 4)
    hold.engage([0.9] * 4)
    assert hold.anchor == [1.0] * 4, hold.anchor
    second = hold.update([0.9] * 4, [0.0] * 4)
    assert second > first > 0.0, (first, second)
    return {"anchor": hold.anchor, "first": first, "second": second}


def release_and_inactive_zero() -> Dict[str, Any]:
    fresh = BrakeWheelHold()
    assert fresh.update([1.0] * 4, [1.0] * 4) == 0.0
    assert fresh.active is False and fresh.anchor is None
    hold = BrakeWheelHold()
    hold.engage([1.0] * 4)
    settle(hold, [0.5] * 4, [0.0] * 4, 5)
    hold.release()
    assert hold.active is False
    assert hold.anchor is None
    assert hold.last_command == 0.0
    assert hold.debug == {}
    assert hold.update([0.5] * 4, [0.0] * 4) == 0.0
    return {"inactive_command": 0.0}


def no_hidden_unit_scaling() -> Dict[str, Any]:
    """Guard against the rad/s -> normalized-action unit confusion.

    This module's ceiling is 0.6 rad/s physical.  Under a schema scale of 5 that
    is 0.12 normalized, and if applied upstream of a wheel gain of 8 it is
    0.015.  The module must emit 0.6 and must never pre-multiply by 8 (4.8) nor
    hand back an already-normalized 0.12 / 0.015.
    """
    hold = BrakeWheelHold()
    hold.engage([3.0] * 4)
    cmd = settle(hold, [0.0] * 4, [0.0] * 4, 300)
    assert abs(cmd - 0.6) < 1e-9, cmd
    for forbidden in (0.6 * 8.0, 0.6 / 5.0, 0.6 / 5.0 / 8.0):
        assert abs(cmd - forbidden) > 1e-6, (cmd, forbidden)
    described = hold.describe()
    assert described["output"]["range_rad_s"] == [-0.6, 0.6]
    json.dumps(described)
    json.dumps(hold.debug)
    return {
        "command_rad_s": cmd,
        "normalized_scale5": cmd / 5.0,
        "normalized_scale5_pre_gain8": cmd / 5.0 / 8.0,
        "forbidden_pre_multiplied": 0.6 * 8.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU audit for BrakeWheelHold")
    parser.add_argument("--output", default="results/task_b_brake_wheel_hold_cpu_audit.json")
    args = parser.parse_args()

    check("error_sign_positive", sign_positive)
    check("error_sign_negative", sign_negative)
    check("wrap_equivalence_2pi_4pi", wrap_equivalence)
    check("speed_cap", speed_cap)
    check("slew_per_step", slew_per_step)
    check("bad_inputs_rejected", bad_inputs)
    check("bad_parameters_rejected", bad_parameters)
    check("engage_copies_input", engage_copies_input)
    check("repeat_engage_does_not_relatch", repeat_engage_does_not_relatch)
    check("release_and_inactive_zero", release_and_inactive_zero)
    check("no_hidden_unit_scaling", no_hidden_unit_scaling)

    passed = sum(1 for c in CHECKS if c["passed"])
    report = {
        "module": "task_b.brake_wheel_hold",
        "device": "cpu",
        "requires_simulator": False,
        "count": len(CHECKS),
        "passed": passed,
        "failed": len(CHECKS) - passed,
        "all_passed": passed == len(CHECKS),
        "checks": CHECKS,
    }

    directory = os.path.dirname(os.path.abspath(args.output))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    for entry in CHECKS:
        status = "PASS" if entry["passed"] else "FAIL"
        extra = "" if entry["passed"] else f" -> {entry['error']}"
        print(f"[{status}] {entry['name']}{extra}")
    print(f"{passed}/{len(CHECKS)} checks passed (cpu); report written to {args.output}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
