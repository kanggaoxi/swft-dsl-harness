#!/usr/bin/env python3
"""Validate one stage and update pipeline state."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from harness_common import (
    DEFAULT_CONFIG,
    load_config,
    load_state,
    now_iso,
    required_output_status,
    run_validation_commands,
    save_json,
    save_state,
    stage_by_id,
    stage_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="harness/work")
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
    stage = stage_by_id(config, stage_id)
    base = stage_dir(workspace, stage_id)

    output_checks = [
        required_output_status(base, rel_path)
        for rel_path in stage.get("required_outputs", [])
    ]
    missing = [item for item in output_checks if not item["exists"]]

    env = os.environ.copy()
    env["HARNESS_WORKSPACE"] = str(workspace)
    env["HARNESS_STAGE"] = stage_id
    command_results = []
    if not missing:
        command_results = run_validation_commands(stage.get("validation_commands", []), cwd=base, env=env)

    commands_passed = all(item["passed"] for item in command_results)
    passed = not missing and commands_passed
    report = {
        "stage": stage_id,
        "validated_at": now_iso(),
        "passed": passed,
        "output_checks": output_checks,
        "missing_required_outputs": missing,
        "command_results": command_results,
    }
    report_path = base / "validation" / "VALIDATION_REPORT.json"
    save_json(report_path, report)

    stage_state = state["stages"][stage_id]
    stage_state["status"] = "passed" if passed else "failed"
    stage_state["last_validation_report"] = str(report_path.resolve())
    stage_state["validated_at"] = report["validated_at"] if passed else None
    state["updated_at"] = now_iso()
    save_state(workspace, state)

    if passed:
        print(f"stage {stage_id} passed")
        print(f"report: {report_path}")
        return 0

    print(f"stage {stage_id} failed")
    if missing:
        print("missing required outputs:")
        for item in missing:
            print(f"  - {item['path']}")
    failed_commands = [item for item in command_results if not item["passed"]]
    if failed_commands:
        print("failed validation commands:")
        for item in failed_commands:
            print(f"  - {item['command']} -> {item['returncode']}")
    print(f"report: {report_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
