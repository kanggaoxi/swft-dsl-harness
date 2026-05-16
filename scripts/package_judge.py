#!/usr/bin/env python3
"""Generate an isolated judge package for one completed pipeline stage."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from harness_common import (
    DEFAULT_CONFIG,
    load_config,
    load_state,
    now_iso,
    required_output_status,
    resolve_input_ref,
    save_json,
    save_state,
    stage_by_id,
    stage_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="work")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def render_judge_task(stage: dict, manifest: dict, precision: dict) -> str:
    if "weight_dtype_policy" in precision:
        precision_lines = [
            f"- weight dtype policy: `{precision.get('weight_dtype_policy', 'unspecified')}`",
            f"- model input dtype policy: `{precision.get('model_input_dtype_policy', 'unspecified')}`",
            f"- golden dtype policy: `{precision.get('golden_dtype_policy', 'unspecified')}`",
            f"- DSL model IO dtype policy: `{precision.get('dsl_model_io_dtype_policy', 'unspecified')}`",
            f"- final comparison: `{precision.get('final_comparison_metric', 'unspecified')}` <= `{precision.get('final_comparison_rtol', 'unspecified')}`",
            f"- partition comparison default: `{precision.get('partition_comparison_metric', 'unspecified')}` <= `{precision.get('partition_comparison_default_rtol', 'unspecified')}`",
            f"- partition documented override allowed: `{precision.get('partition_allow_documented_override', False)}`",
        ]
    else:
        precision_lines = [
            f"- model entry inputs and weights: `{precision.get('model_input_dtype', 'unspecified')}`",
            f"- torch reference and golden outputs: `{precision.get('torch_reference_dtype', 'unspecified')}`",
            f"- SWFT DSL runtime: `{precision.get('dsl_runtime_dtype', 'unspecified')}`",
            f"- comparison metric: `{precision.get('comparison_metric', 'unspecified')}`",
            f"- required tolerance: `{precision.get('comparison_rtol', 'unspecified')}`",
        ]
    lines = [
        f"# Judge Task: {stage['id']} - {stage['name']}",
        "",
        "## Role",
        "",
        "You are an independent judge agent. You must not inherit or rely on the worker agent's conversation context.",
        "Evaluate the stage only from this package, the stage outputs, and the referenced input artifacts.",
        "",
        "## Objective",
        "",
        "Decide whether this stage is genuinely correct enough for the next stage to consume.",
        "Do not implement missing work. Report precise findings and required fixes.",
        "",
        "## Precision Contract",
        "",
        *precision_lines,
        "",
        "## What To Read",
        "",
        "1. `JUDGE_INPUT_MANIFEST.json` for all available paths.",
        "2. `JUDGE_GUIDE.md` for general judging rules.",
        "3. The stage outputs listed in `stage_output_checks`.",
        "4. The stage-specific checklist below.",
        "",
        "## Stage-Specific Checklist",
        "",
    ]
    checklist = stage.get("judge_checklist", [])
    if checklist:
        for item in checklist:
            lines.append(f"- {item}")
    else:
        lines.append("- Check that required outputs are internally consistent and usable by the next stage.")
    lines.extend([
        "",
        "## Required Judge Output",
        "",
        "Write exactly this file under the harness workspace:",
        "",
        f"- `{manifest['judge_report']}`",
        "",
        "The JSON must contain:",
        "",
        "```json",
        "{",
        '  "stage": "stage_id",',
        '  "passed": false,',
        '  "reviewed_files": [],',
        '  "checked_items": [],',
        '  "findings": [],',
        '  "required_fixes": []',
        "}",
        "```",
        "",
        "Set `passed` to true only when the stage can safely advance.",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    state = load_state(workspace)
    config_path = Path(args.config).resolve() if args.config else Path(state.get("pipeline_config", DEFAULT_CONFIG)).resolve()
    config = load_config(config_path)
    stage_id = args.stage or state["current_stage"]
    stage = stage_by_id(config, stage_id)
    stage_state = state["stages"][stage_id]
    if stage_state.get("status") != "passed" and not args.force:
        raise SystemExit(f"stage {stage_id} has not passed mechanical validation; use --force to package judge anyway")

    base = stage_dir(workspace, stage_id)
    package = base / "judge_package"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True, exist_ok=True)
    judge_dir = base / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    input_refs = [resolve_input_ref(config, ref, workspace) for ref in stage.get("input_refs", [])]
    output_checks = [required_output_status(base, rel_path) for rel_path in stage.get("required_outputs", [])]
    validation_report = base / "validation" / "VALIDATION_REPORT.json"
    guide = (Path(__file__).resolve().parents[1] / "docs" / "JUDGE_GUIDE-CH.md").resolve()
    manifest = {
        "stage": stage_id,
        "created_at": now_iso(),
        "workspace": str(workspace),
        "precision": config.get("precision", {}),
        "stage_objective": stage.get("objective", ""),
        "stage_input_refs": input_refs,
        "stage_output_checks": output_checks,
        "mechanical_validation_report": str(validation_report.resolve()),
        "mechanical_validation_exists": validation_report.exists(),
        "judge_guide": str(guide),
        "judge_report": str((judge_dir / "JUDGE_REPORT.json").resolve()),
    }
    save_json(package / "JUDGE_INPUT_MANIFEST.json", manifest)
    (package / "JUDGE_TASK.md").write_text(
        render_judge_task(stage, manifest, config.get("precision", {})),
        encoding="utf-8",
    )
    if guide.exists():
        shutil.copy2(guide, package / "JUDGE_GUIDE.md")

    stage_state["judge_package_path"] = str(package.resolve())
    state["updated_at"] = now_iso()
    save_state(workspace, state)

    print(f"packaged judge for {stage_id}: {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
