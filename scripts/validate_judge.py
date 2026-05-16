#!/usr/bin/env python3
"""Validate an independent judge report and update pipeline state."""

from __future__ import annotations

import argparse
from pathlib import Path

from harness_common import DEFAULT_CONFIG, load_config, load_json, load_state, now_iso, save_json, save_state, stage_by_id, stage_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="work")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    state = load_state(workspace)
    config_path = Path(args.config).resolve() if args.config else Path(state.get("pipeline_config", DEFAULT_CONFIG)).resolve()
    config = load_config(config_path)
    stage_id = args.stage or state["current_stage"]
    stage_by_id(config, stage_id)
    base = stage_dir(workspace, stage_id)
    report_path = base / "judge" / "JUDGE_REPORT.json"
    validation_path = base / "judge" / "JUDGE_VALIDATION_REPORT.json"
    mechanical_report_path = base / "validation" / "VALIDATION_REPORT.json"

    errors: list[str] = []
    stage_state = state["stages"][stage_id]
    mechanical_report = None
    if stage_state.get("status") != "passed":
        errors.append(f"stage {stage_id} has not passed mechanical validation")
    if not mechanical_report_path.exists():
        errors.append(f"missing mechanical validation report: {mechanical_report_path}")
    else:
        try:
            mechanical_report = load_json(mechanical_report_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cannot parse mechanical validation report: {exc}")
    if isinstance(mechanical_report, dict) and mechanical_report.get("passed") is not True:
        errors.append("mechanical validation report is not passed")

    report = None
    if not report_path.exists():
        errors.append(f"missing judge report: {report_path}")
    else:
        try:
            report = load_json(report_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cannot parse judge report: {exc}")

    if isinstance(report, dict):
        if report.get("stage") not in (stage_id, None):
            errors.append(f"judge report stage mismatch: {report.get('stage')} != {stage_id}")
        if not isinstance(report.get("passed"), bool):
            errors.append("judge report must contain boolean field `passed`")
    passed = isinstance(report, dict) and not errors and report.get("passed") is True
    if isinstance(report, dict) and report.get("passed") is False:
        findings = report.get("findings", [])
        required_fixes = report.get("required_fixes", [])
    else:
        findings = []
        required_fixes = []

    validation = {
        "stage": stage_id,
        "validated_at": now_iso(),
        "passed": passed,
        "errors": errors,
        "mechanical_validation_report": str(mechanical_report_path.resolve()),
        "mechanical_validation_passed": mechanical_report.get("passed") if isinstance(mechanical_report, dict) else None,
        "judge_report": str(report_path.resolve()),
        "judge_passed_field": report.get("passed") if isinstance(report, dict) else None,
        "findings": findings,
        "required_fixes": required_fixes,
    }
    save_json(validation_path, validation)

    stage_state["last_judge_report"] = str(report_path.resolve())
    stage_state["judge_validation_report"] = str(validation_path.resolve())
    stage_state["judge_passed"] = passed
    stage_state["judged_at"] = validation["validated_at"] if passed else None
    if not passed:
        stage_state["status"] = "failed"
    state["updated_at"] = now_iso()
    save_state(workspace, state)

    if passed:
        print(f"judge passed for stage {stage_id}")
        print(f"report: {validation_path}")
        return 0

    print(f"judge failed for stage {stage_id}")
    for error in errors:
        print(f"  - {error}")
    for fix in required_fixes:
        print(f"  - required fix: {fix}")
    print(f"report: {validation_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
