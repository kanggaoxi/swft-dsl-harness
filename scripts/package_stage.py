#!/usr/bin/env python3
"""Generate an isolated task package for one pipeline stage."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from harness_common import (
    DEFAULT_CONFIG,
    copy_input,
    load_config,
    load_state,
    now_iso,
    previous_stage_id,
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
    parser.add_argument("--copy-inputs", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def append_numbered(lines: list[str], title: str, items: list[str]) -> None:
    if not items:
        return
    lines.extend(["", f"## {title}", ""])
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item}")


def append_bullets(lines: list[str], title: str, items: list[str]) -> None:
    if not items:
        return
    lines.extend(["", f"## {title}", ""])
    for item in items:
        lines.append(f"- {item}")


def render_task(stage: dict, input_manifest: dict, output_contract: dict, validation_text: str, precision: dict | None = None) -> str:
    lines = [
        f"# Agent Task: {stage['id']} - {stage['name']}",
        "",
        "## Role",
        "",
        stage["agent_role"],
        "",
        "You are a bounded stage agent. Complete only this stage. Do not reinterpret the entire pipeline.",
        "",
        "## Objective",
        "",
        stage["objective"],
        "",
        "## Precision Contract",
        "",
    ]
    if precision:
        if "weight_dtype_policy" in precision:
            lines.extend([
                f"- weight dtype policy: `{precision.get('weight_dtype_policy', 'unspecified')}`",
                f"- model input dtype policy: `{precision.get('model_input_dtype_policy', 'unspecified')}`",
                f"- golden dtype policy: `{precision.get('golden_dtype_policy', 'unspecified')}`",
                f"- DSL model IO dtype policy: `{precision.get('dsl_model_io_dtype_policy', 'unspecified')}`",
                f"- final comparison: `{precision.get('final_comparison_metric', 'unspecified')}` <= `{precision.get('final_comparison_rtol', 'unspecified')}`",
                f"- partition comparison default: `{precision.get('partition_comparison_metric', 'unspecified')}` <= `{precision.get('partition_comparison_default_rtol', 'unspecified')}`",
                f"- partition documented override allowed: `{precision.get('partition_allow_documented_override', False)}`",
                "",
            ])
        else:
            lines.extend([
                f"- model entry inputs and weights: `{precision.get('model_input_dtype', 'unspecified')}`",
                f"- torch reference and golden outputs: `{precision.get('torch_reference_dtype', 'unspecified')}`",
                f"- SWFT DSL runtime: `{precision.get('dsl_runtime_dtype', 'unspecified')}`",
                f"- comparison metric: `{precision.get('comparison_metric', 'unspecified')}`",
                f"- required tolerance: `{precision.get('comparison_rtol', 'unspecified')}`",
                "",
            ])
    else:
        lines.extend(["No precision contract is configured for this pipeline.", ""])
    lines.extend([
        "## Inputs",
        "",
        "Read `INPUT_MANIFEST.json` first. It lists the exact files available to this stage.",
        "",
    ])
    missing = [item for item in input_manifest["inputs"] if not item["exists"]]
    if missing:
        lines.extend([
            "The following declared inputs are currently missing. If they are required for your task, stop and report the missing paths:",
            "",
        ])
        for item in missing:
            lines.append(f"- `{item['label']}`: `{item['path']}`")
        lines.append("")
    existing = [item for item in input_manifest["inputs"] if item["exists"]]
    if existing:
        lines.extend([
            "Available input paths:",
            "",
        ])
        for item in existing:
            lines.append(f"- `{item['label']}`: `{item['path']}`")
        lines.append("")
    append_numbered(lines, "Procedure", stage.get("procedure", []))
    append_bullets(lines, "Quality Checks Before Handoff", stage.get("quality_checks", []))
    lines.extend([
        "## Allowed Edits",
        "",
        "Edit only these paths relative to the harness workspace:",
        "",
    ])
    for path in stage.get("allowed_edits", []):
        lines.append(f"- `{path}`")
    if stage.get("subagent_packages", {}).get("enabled"):
        lines.extend([
            "",
            "## Subagent Coordination",
            "",
            "This stage is expected to be coordinated by a main agent. The main agent should create one bounded task package per partition, then assign those packages to subagents with disjoint ownership.",
            "",
            "Generate partition subagent packages from the repository root with:",
            "",
            "```bash",
            f"python3 {stage['subagent_packages']['script']} --workspace work",
            "```",
            "",
            "Each subagent package contains a single `partition.json`, the shared manifests, and a strict output contract. Subagents should not modify files owned by other partitions.",
        ])
    lines.extend([
        "",
        "Do not modify upstream stage outputs, golden files, pipeline config, or harness scripts unless this task explicitly says so.",
        "",
        "## Required Outputs",
        "",
        "Write all deliverables under this stage's `output/` directory unless the contract says otherwise.",
        "",
    ])
    for path in output_contract["required_outputs"]:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## Validation",
        "",
        validation_text,
        "",
        "## Completion Response",
        "",
        "Report only: files changed, validation commands run, pass/fail status, and unresolved issues.",
        "",
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

    prev_id = previous_stage_id(config, stage_id)
    if prev_id and state["stages"][prev_id]["status"] != "passed" and not args.force:
        raise SystemExit(f"previous stage {prev_id} is not passed; use --force to package anyway")

    base = stage_dir(workspace, stage_id)
    package = base / "agent_package"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True, exist_ok=True)

    inputs = []
    for ref in stage.get("input_refs", []):
        item = resolve_input_ref(config, ref, workspace)
        if args.copy_inputs:
            copied = copy_input(Path(item["path"]), package / "files")
            item["copied_path"] = copied
        inputs.append(item)

    input_manifest = {
        "stage": stage_id,
        "created_at": now_iso(),
        "workspace": str(workspace),
        "precision": config.get("precision", {}),
        "inputs": inputs,
    }
    output_contract = {
        "stage": stage_id,
        "required_outputs": stage.get("required_outputs", []),
        "allowed_edits": stage.get("allowed_edits", []),
    }
    validation_text = render_validation(stage)

    save_json(package / "INPUT_MANIFEST.json", input_manifest)
    save_json(package / "OUTPUT_CONTRACT.json", output_contract)
    (package / "VALIDATION.md").write_text(validation_text, encoding="utf-8")
    (package / "AGENT_TASK.md").write_text(
        render_task(stage, input_manifest, output_contract, validation_text, config.get("precision", {})),
        encoding="utf-8",
    )

    stage_state["status"] = "packaged"
    stage_state["attempts"] = int(stage_state.get("attempts", 0)) + 1
    stage_state["package_path"] = str(package.resolve())
    state["updated_at"] = now_iso()
    save_state(workspace, state)

    print(f"packaged stage {stage_id}: {package}")
    return 0


def render_validation(stage: dict) -> str:
    lines = [
        "After implementation, run the orchestrator validation gate from the repository root:",
        "",
        f"```bash\npython3 scripts/validate_stage.py --workspace work --stage {stage['id']}\n```",
        "",
        "The mechanical gate checks required output paths and any configured validation commands.",
    ]
    commands = stage.get("validation_commands", [])
    if commands:
        lines.extend(["", "Configured validation commands:"])
        for command in commands:
            lines.append(f"- `{command}`")
    else:
        lines.extend(["", "No stage-specific validation commands are configured yet. Required output files are still checked."])
    lines.extend([
        "",
        "After the mechanical gate passes, create an independent judge package:",
        "",
        f"```bash\npython3 scripts/package_judge.py --workspace work --stage {stage['id']}\n```",
        "",
        "Give `stages/"
        f"{stage['id']}/judge_package/` to a fresh judge agent that has not inherited this worker agent's context.",
        "The judge must write `stages/"
        f"{stage['id']}/judge/JUDGE_REPORT.json`.",
        "",
        "Then run:",
        "",
        f"```bash\npython3 scripts/validate_judge.py --workspace work --stage {stage['id']}\n```",
        "",
        "Only after both validation and judge pass should the pipeline advance.",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
