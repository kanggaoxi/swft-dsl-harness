#!/usr/bin/env python3
"""Check configured external input paths for a harness workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

from harness_common import (
    DEFAULT_CONFIG,
    input_paths_path,
    load_config,
    load_input_paths,
    resolve_external,
    save_json,
    stage_by_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="work")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", default=None, help="Check only inputs required by this stage.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    config = load_config(Path(args.config).resolve())
    stage_id = args.stage
    if stage_id:
        stage_by_id(config, stage_id)

    overrides = load_input_paths(workspace)
    known_labels = {item["label"] for item in config.get("external_inputs", [])}
    unknown_labels = sorted(label for label in overrides if label not in known_labels)

    checks = []
    missing_required = []
    for item in config.get("external_inputs", []):
        required_for = item.get("required_for", [])
        required_now = not stage_id or stage_id in required_for
        resolved = resolve_external(config, item["label"], workspace)
        path = Path(resolved["resolved_path"])
        check = {
            "label": item["label"],
            "path": str(path),
            "exists": path.exists(),
            "required_for": required_for,
            "required_now": required_now,
            "path_source": resolved.get("path_source"),
            "configured_path": resolved.get("path"),
            "default_path": resolved.get("default_path"),
            "relative_to": resolved.get("relative_to"),
        }
        checks.append(check)
        if required_now and not check["exists"]:
            missing_required.append(check)

    report = {
        "workspace": str(workspace),
        "input_paths_file": str(input_paths_path(workspace)),
        "stage": stage_id,
        "passed": not unknown_labels and not missing_required,
        "unknown_labels": unknown_labels,
        "missing_required": missing_required,
        "checks": checks,
    }
    report_path = workspace / "input_check_report.json"
    save_json(report_path, report)

    if unknown_labels:
        print("unknown labels in input_paths.json:")
        for label in unknown_labels:
            print(f"  - {label}")
    if missing_required:
        print("missing required inputs:")
        for item in missing_required:
            print(f"  - {item['label']}: {item['path']}")
    for item in checks:
        marker = "OK" if item["exists"] else "MISSING"
        scope = "required" if item["required_now"] else "optional"
        print(f"{marker} [{scope}] {item['label']}: {item['path']}")
    print(f"report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
