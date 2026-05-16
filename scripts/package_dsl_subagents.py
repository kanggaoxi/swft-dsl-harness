#!/usr/bin/env python3
"""Create one isolated DSL implementation package per graph partition."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

from harness_common import (
    DEFAULT_CONFIG,
    load_config,
    load_json,
    load_state,
    now_iso,
    resolve_input_ref,
    save_json,
    stage_by_id,
    stage_dir,
)


DEFAULT_STAGE = "05_dsl_partitions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="work")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--partitions", default=None, help="Comma-separated partition ids to package. Defaults to all.")
    parser.add_argument("--ready-only", action="store_true", help="Package only partitions with no unresolved implementation dependencies.")
    parser.add_argument("--clean", action="store_true", help="Remove existing subagent package directory first.")
    return parser.parse_args()


def sanitize_id(value: Any, fallback: str) -> str:
    raw = str(value if value not in (None, "") else fallback)
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return raw or fallback


def extract_partitions(plan: Any) -> list[dict[str, Any]]:
    if isinstance(plan, list):
        partitions = plan
    elif isinstance(plan, dict):
        for key in ("partitions", "subgraphs", "segments"):
            if isinstance(plan.get(key), list):
                partitions = plan[key]
                break
        else:
            raise ValueError("partition plan must contain a list under one of: partitions, subgraphs, segments")
    else:
        raise TypeError("partition plan must be a JSON object or list")

    normalized = []
    for idx, item in enumerate(partitions):
        if not isinstance(item, dict):
            raise TypeError(f"partition at index {idx} is not an object")
        partition_id = sanitize_id(
            item.get("id", item.get("partition_id", item.get("name"))),
            f"partition_{idx:03d}",
        )
        normalized.append({"id": partition_id, "index": idx, "spec": item})
    return normalized


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def dependency_info(partition: dict[str, Any]) -> dict[str, Any]:
    spec = partition["spec"]
    semantic_deps = listify(spec.get("semantic_deps", spec.get("dependencies", spec.get("deps"))))
    implementation_deps = listify(spec.get("implementation_deps", spec.get("impl_deps")))
    fusion_group = spec.get("fusion_group")
    independent = spec.get("can_implement_independently")
    if independent is None:
        independent = not implementation_deps
    blocked_reasons = []
    if implementation_deps:
        blocked_reasons.append(f"implementation_deps={implementation_deps}")
    if independent is False:
        blocked_reasons.append("can_implement_independently=false")
    return {
        "semantic_deps": semantic_deps,
        "implementation_deps": implementation_deps,
        "fusion_group": fusion_group,
        "can_implement_independently": bool(independent),
        "ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
    }


def render_agent_task(partition: dict[str, Any], paths: dict[str, str], precision: dict[str, Any], dep_info: dict[str, Any]) -> str:
    partition_id = partition["id"]
    if "weight_dtype_policy" in precision:
        accuracy_lines = [
            f"- weight dtype policy: `{precision.get('weight_dtype_policy', 'unspecified')}`",
            f"- model input dtype policy: `{precision.get('model_input_dtype_policy', 'unspecified')}`",
            f"- golden dtype policy: `{precision.get('golden_dtype_policy', 'unspecified')}`",
            f"- DSL model IO dtype policy: `{precision.get('dsl_model_io_dtype_policy', 'unspecified')}`",
            f"- final comparison: `{precision.get('final_comparison_metric', 'unspecified')}` <= `{precision.get('final_comparison_rtol', 'unspecified')}`",
            f"- partition comparison default: `{precision.get('partition_comparison_metric', 'unspecified')}` <= `{precision.get('partition_comparison_default_rtol', 'unspecified')}`",
            f"- partition documented override allowed: `{precision.get('partition_allow_documented_override', False)}`",
        ]
    else:
        accuracy_lines = [
            f"- model entry inputs and weights: `{precision.get('model_input_dtype', 'unspecified')}`",
            f"- torch reference and golden outputs: `{precision.get('torch_reference_dtype', 'unspecified')}`",
            f"- SWFT DSL runtime: `{precision.get('dsl_runtime_dtype', 'unspecified')}`",
            f"- comparison metric: `{precision.get('comparison_metric', 'unspecified')}`",
            f"- required tolerance: `{precision.get('comparison_rtol', 'unspecified')}`",
        ]
    return "\n".join([
        f"# Subagent Task: Implement DSL for `{partition_id}`",
        "",
        "## Role",
        "",
        "You are a bounded DSL partition implementation subagent. You are not alone in this stage: other subagents may implement different partitions in parallel.",
        "",
        "## Objective",
        "",
        f"Implement only partition `{partition_id}` in SWFT DSL and validate it against the torch-generated golden bins.",
        "",
        "Do not modify other partitions. Do not change the global partition plan, golden files, harness scripts, or upstream artifacts.",
        "",
        "## Required Inputs",
        "",
        "Read these files from this package first:",
        "",
        "- `partition.json`: the only partition you own.",
        "- `INPUT_MANIFEST.json`: shared paths for model IR, partition plan, golden manifest, cases, skeleton DSL, similar DSL, SWFT flow docs, and implementation notes.",
        "- `OUTPUT_CONTRACT.json`: files this subagent must produce.",
        "",
        "## Dependency Status",
        "",
        f"- semantic deps: `{dep_info['semantic_deps']}`",
        f"- implementation deps: `{dep_info['implementation_deps']}`",
        f"- fusion group: `{dep_info['fusion_group']}`",
        f"- can implement independently: `{dep_info['can_implement_independently']}`",
        "",
        "Semantic deps describe graph dataflow and do not block isolated partition development when torch-captured partition inputs are available.",
        "Implementation deps describe layout, fusion, or shared code dependencies and should be resolved by the main agent before this package is assigned.",
        "If `can implement independently` is false or implementation deps are non-empty, stop and report the blocked status unless the main agent explicitly assigned this package together with its dependency or fusion group.",
        "",
        "## Accuracy Target",
        "",
        *accuracy_lines,
        "",
        "## Allowed Output Area",
        "",
        f"Write implementation artifacts only under `{paths['output_dir']}` and logs only under `{paths['log_dir']}`.",
        "",
        "## Procedure",
        "",
        "1. Inspect `partition.json` for operation sequence, inputs, outputs, shapes, dtypes, and dependencies.",
        "2. Read the SWFT flow docs and implementation notes from `INPUT_MANIFEST.json` before writing DSL.",
        "3. Reuse patterns from the similar DSL implementation where applicable.",
        "4. Implement the partition in this partition-owned file: `implementation.py`.",
        "5. Use `slice_to_ub` for GM reads and `insert_to_gm` for GM writes unless you document a reason not to.",
        "6. Compile and run only this partition's test case.",
        "7. Compare actual output against the golden bins recorded in `golden_manifest.json`.",
        "8. Write the required output contract files.",
        "",
        "## Completion Response",
        "",
        "Report only files changed, validation commands run, pass/fail status, and unresolved issues.",
        "",
    ])


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    state = load_state(workspace)
    config_path = Path(args.config).resolve() if args.config else Path(state.get("pipeline_config", DEFAULT_CONFIG)).resolve()
    config = load_config(config_path)
    stage = stage_by_id(config, args.stage)
    subagent_cfg = stage.get("subagent_packages", {})
    if not subagent_cfg.get("enabled"):
        raise SystemExit(f"stage {args.stage} does not enable subagent packages")

    stage_base = stage_dir(workspace, args.stage)
    package_dir = workspace / subagent_cfg.get("package_dir", f"stages/{args.stage}/subagent_packages")
    if args.clean and package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    partition_plan_ref = subagent_cfg.get("partition_plan", "stage:02_partition/output/partition_plan.json")
    partition_plan_path = Path(resolve_input_ref(config, partition_plan_ref, workspace)["path"])
    if not partition_plan_path.exists():
        raise SystemExit(f"partition plan does not exist: {partition_plan_path}")
    partition_plan = load_json(partition_plan_path)
    partitions = extract_partitions(partition_plan)
    dep_by_id = {item["id"]: dependency_info(item) for item in partitions}
    ready_partitions = [item for item in partitions if dep_by_id[item["id"]]["ready"]]
    blocked_partitions = [item for item in partitions if not dep_by_id[item["id"]]["ready"]]

    requested = None
    if args.partitions:
        requested = {sanitize_id(item.strip(), item.strip()) for item in args.partitions.split(",") if item.strip()}
        partitions = [item for item in partitions if item["id"] in requested]
        missing = sorted(requested - {item["id"] for item in partitions})
        if missing:
            raise SystemExit(f"requested partitions not found: {', '.join(missing)}")
    elif args.ready_only:
        partitions = ready_partitions

    shared_inputs = []
    for ref in stage.get("input_refs", []):
        shared_inputs.append(resolve_input_ref(config, ref, workspace))

    created = []
    for partition in partitions:
        partition_id = partition["id"]
        base = package_dir / partition_id
        if base.exists():
            shutil.rmtree(base)
        output_dir = stage_base / "output" / "partitions" / partition_id
        log_dir = stage_base / "logs" / "partitions" / partition_id
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        base.mkdir(parents=True, exist_ok=True)

        dep_info = dep_by_id[partition_id]
        save_json(base / "partition.json", {
            "id": partition_id,
            "index": partition["index"],
            "dependency_info": dep_info,
            "spec": partition["spec"],
        })
        paths = {
            "output_dir": str(output_dir.resolve()),
            "log_dir": str(log_dir.resolve()),
        }
        input_manifest = {
            "stage": args.stage,
            "partition_id": partition_id,
            "created_at": now_iso(),
            "workspace": str(workspace),
            "precision": config.get("precision", {}),
            "paths": paths,
            "dependency_info": dep_info,
            "shared_inputs": shared_inputs,
        }
        output_contract = {
            "stage": args.stage,
            "partition_id": partition_id,
            "required_outputs": [
                f"{paths['output_dir']}/implementation.py",
                f"{paths['output_dir']}/correctness_report.json",
                f"{paths['output_dir']}/validation_notes.md",
            ],
        }
        save_json(base / "INPUT_MANIFEST.json", input_manifest)
        save_json(base / "OUTPUT_CONTRACT.json", output_contract)
        (base / "AGENT_TASK.md").write_text(
            render_agent_task(partition, paths, config.get("precision", {}), dep_info),
            encoding="utf-8",
        )
        created.append({
            "partition_id": partition_id,
            **dep_info,
            "package_path": str(base.resolve()),
            "output_dir": paths["output_dir"],
            "log_dir": paths["log_dir"],
        })

    blocked = []
    for partition in blocked_partitions:
        info = dep_by_id[partition["id"]]
        blocked.append({
            "partition_id": partition["id"],
            "index": partition["index"],
            **info,
        })

    manifest = {
        "stage": args.stage,
        "created_at": now_iso(),
        "package_dir": str(package_dir.resolve()),
        "launch_mode": "manual_sessions",
        "ready_only": args.ready_only,
        "partition_count": len(created),
        "ready_partition_count": len([item for item in created if item["ready"]]),
        "blocked_partition_count": len(blocked),
        "partitions": created,
        "ready_partitions": [item for item in created if item["ready"]],
        "blocked_partitions": blocked,
        "manual_launch_instructions": [
            "Open one fresh agent session per package_path that you want to run.",
            "Give that agent only the package directory path and ask it to follow AGENT_TASK.md.",
            "Do not let subagents edit shared target_dsl files or other partition output directories.",
            "After subagents finish, the stage 05 main agent reviews their outputs and writes the aggregate manifests."
        ],
    }
    manifest_path = stage_base / "output" / "subagent_task_manifest.json"
    save_json(manifest_path, manifest)
    print(f"created {len(created)} subagent package(s): {package_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
