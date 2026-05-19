#!/usr/bin/env python3
"""Create isolated DSL implementation and judge packages for graph partition bundles."""

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
DEFAULT_MAX_BUNDLE_PARTITIONS = 4
ELEMENTWISE_HINTS = {
    "abs",
    "add",
    "cast",
    "div",
    "elementwise",
    "elu",
    "exp",
    "gelu",
    "maximum",
    "minimum",
    "mul",
    "neg",
    "pow",
    "relu",
    "rsqrt",
    "sigmoid",
    "sqrt",
    "sub",
    "tanh",
}
HEAVY_OP_HINTS = {
    "argmax",
    "batchmatmul",
    "concat",
    "conv",
    "gather",
    "layernorm",
    "matmul",
    "reduce",
    "softmax",
    "sort",
    "topk",
    "transpose",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="work")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--partitions", default=None, help="Comma-separated partition ids to package. Defaults to all.")
    parser.add_argument("--ready-only", action="store_true", help="Package only bundles with no unresolved implementation dependencies.")
    parser.add_argument("--max-bundle-partitions", type=int, default=DEFAULT_MAX_BUNDLE_PARTITIONS, help="Maximum number of small independent partitions to merge into one bundle.")
    parser.add_argument("--no-small-bundles", action="store_true", help="Disable bundling of adjacent small independent elementwise partitions.")
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
        "partition_level_ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
    }


def all_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join([str(k) for k in value.keys()] + [all_text(v) for v in value.values()])
    if isinstance(value, list):
        return " ".join(all_text(item) for item in value)
    return str(value)


def is_small_elementwise(partition: dict[str, Any], dep_info: dict[str, Any]) -> bool:
    if dep_info["implementation_deps"] or dep_info["fusion_group"] or not dep_info["can_implement_independently"]:
        return False
    text = all_text(partition["spec"]).lower()
    if any(hint in text for hint in HEAVY_OP_HINTS):
        return False
    return any(hint in text for hint in ELEMENTWISE_HINTS)


class DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def forced_partition_groups(partitions: list[dict[str, Any]], dep_by_id: dict[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ids = [item["id"] for item in partitions]
    id_set = set(ids)
    dsu = DisjointSet(ids)
    by_fusion_group: dict[str, list[str]] = {}

    for partition in partitions:
        partition_id = partition["id"]
        info = dep_by_id[partition_id]
        fusion_group = info.get("fusion_group")
        if fusion_group not in (None, ""):
            by_fusion_group.setdefault(str(fusion_group), []).append(partition_id)
        for dep in info["implementation_deps"]:
            if dep in id_set:
                dsu.union(partition_id, dep)
        if not info["can_implement_independently"]:
            for dep in info["semantic_deps"]:
                if dep in id_set:
                    dsu.union(partition_id, dep)

    for group_ids in by_fusion_group.values():
        head = group_ids[0]
        for partition_id in group_ids[1:]:
            dsu.union(head, partition_id)

    groups: dict[str, list[dict[str, Any]]] = {}
    for partition in partitions:
        groups.setdefault(dsu.find(partition["id"]), []).append(partition)
    return [sorted(group, key=lambda item: item["index"]) for group in groups.values()]


def bundle_id_for(partitions: list[dict[str, Any]], index: int) -> str:
    if len(partitions) == 1:
        return partitions[0]["id"]
    first = partitions[0]["id"]
    last = partitions[-1]["id"]
    return sanitize_id(f"bundle_{index:03d}_{first}_to_{last}", f"bundle_{index:03d}")


def bundle_status(bundle_partitions: list[dict[str, Any]], dep_by_id: dict[str, dict[str, Any]], id_set: set[str]) -> dict[str, Any]:
    bundle_ids = {item["id"] for item in bundle_partitions}
    reasons = []
    semantic_deps = []
    implementation_deps = []
    fusion_groups = []
    small_elementwise = []
    for partition in bundle_partitions:
        partition_id = partition["id"]
        info = dep_by_id[partition_id]
        semantic_deps.extend(info["semantic_deps"])
        implementation_deps.extend(info["implementation_deps"])
        if info["fusion_group"] not in (None, ""):
            fusion_groups.append(str(info["fusion_group"]))
        if is_small_elementwise(partition, info):
            small_elementwise.append(partition_id)
        for dep in info["implementation_deps"]:
            if dep not in id_set:
                reasons.append(f"{partition_id} implementation_dep {dep} is not in the selected partition plan")
            elif dep not in bundle_ids:
                reasons.append(f"{partition_id} implementation_dep {dep} is outside this bundle")
        if not info["can_implement_independently"] and len(bundle_ids) == 1:
            reasons.append(f"{partition_id} has can_implement_independently=false and was not bundled with a dependency or fusion group")

    return {
        "ready": not reasons,
        "blocked_reasons": sorted(set(reasons)),
        "semantic_deps": sorted(set(semantic_deps)),
        "implementation_deps": sorted(set(implementation_deps)),
        "fusion_groups": sorted(set(fusion_groups)),
        "small_elementwise_partitions": small_elementwise,
    }


def make_bundles(
    partitions: list[dict[str, Any]],
    dep_by_id: dict[str, dict[str, Any]],
    max_bundle_partitions: int,
    merge_small_bundles: bool,
) -> list[dict[str, Any]]:
    id_set = {item["id"] for item in partitions}
    forced_groups = sorted(forced_partition_groups(partitions, dep_by_id), key=lambda group: group[0]["index"])
    raw_bundles: list[list[dict[str, Any]]] = []
    pending_small: list[dict[str, Any]] = []

    def flush_small() -> None:
        nonlocal pending_small
        if pending_small:
            raw_bundles.append(pending_small)
            pending_small = []

    for group in forced_groups:
        is_single_small = (
            merge_small_bundles
            and len(group) == 1
            and max_bundle_partitions > 1
            and is_small_elementwise(group[0], dep_by_id[group[0]["id"]])
        )
        if is_single_small:
            if pending_small and group[0]["index"] != pending_small[-1]["index"] + 1:
                flush_small()
            pending_small.append(group[0])
            if len(pending_small) >= max_bundle_partitions:
                flush_small()
        else:
            flush_small()
            raw_bundles.append(group)
    flush_small()

    bundles = []
    for idx, group in enumerate(raw_bundles):
        status = bundle_status(group, dep_by_id, id_set)
        reason = "small_adjacent_elementwise" if len(group) > 1 and status["small_elementwise_partitions"] else "forced_dependency_or_single_partition"
        bundle_id = bundle_id_for(group, idx)
        bundles.append({
            "bundle_id": bundle_id,
            "index": idx,
            "partition_ids": [item["id"] for item in group],
            "partitions": group,
            "bundle_reason": reason,
            **status,
        })
    return bundles


def accuracy_lines(precision: dict[str, Any]) -> list[str]:
    if "weight_dtype_policy" in precision:
        return [
            f"- weight dtype policy: `{precision.get('weight_dtype_policy', 'unspecified')}`",
            f"- model input dtype policy: `{precision.get('model_input_dtype_policy', 'unspecified')}`",
            f"- golden dtype policy: `{precision.get('golden_dtype_policy', 'unspecified')}`",
            f"- DSL model IO dtype policy: `{precision.get('dsl_model_io_dtype_policy', 'unspecified')}`",
            f"- final comparison: `{precision.get('final_comparison_metric', 'unspecified')}` <= `{precision.get('final_comparison_rtol', 'unspecified')}`",
            f"- partition comparison default: `{precision.get('partition_comparison_metric', 'unspecified')}` <= `{precision.get('partition_comparison_default_rtol', 'unspecified')}`",
            f"- partition documented override allowed: `{precision.get('partition_allow_documented_override', False)}`",
        ]
    return [
        f"- model entry inputs and weights: `{precision.get('model_input_dtype', 'unspecified')}`",
        f"- torch reference and golden outputs: `{precision.get('torch_reference_dtype', 'unspecified')}`",
        f"- SWFT DSL runtime: `{precision.get('dsl_runtime_dtype', 'unspecified')}`",
        f"- comparison metric: `{precision.get('comparison_metric', 'unspecified')}`",
        f"- required tolerance: `{precision.get('comparison_rtol', 'unspecified')}`",
    ]


def render_work_task(bundle: dict[str, Any], paths: dict[str, str], precision: dict[str, Any]) -> str:
    partition_list = ", ".join(f"`{item}`" for item in bundle["partition_ids"])
    return "\n".join([
        f"# Subagent Work Task: Implement DSL Bundle `{bundle['bundle_id']}`",
        "",
        "## Role",
        "",
        "You are a bounded DSL bundle implementation agent. Other agents may implement different bundles in parallel.",
        "",
        "## Objective",
        "",
        f"Implement only these partitions in SWFT DSL and validate them against torch-generated golden bins: {partition_list}.",
        "",
        "Do not modify other bundles. Do not change the global partition plan, golden files, harness scripts, or upstream artifacts.",
        "",
        "## Required Inputs",
        "",
        "Read these files from this package first:",
        "",
        "- `bundle.json`: bundle metadata and dependency status.",
        "- `partitions/*.json`: the only partitions you own.",
        "- `INPUT_MANIFEST.json`: shared paths for model IR, partition plan, golden manifest, cases, skeleton DSL, similar DSL, SWFT flow docs, and implementation notes.",
        "- `OUTPUT_CONTRACT.json`: files this subagent must produce.",
        "",
        "## Bundle Status",
        "",
        f"- ready: `{bundle['ready']}`",
        f"- bundle reason: `{bundle['bundle_reason']}`",
        f"- semantic deps: `{bundle['semantic_deps']}`",
        f"- implementation deps: `{bundle['implementation_deps']}`",
        f"- fusion groups: `{bundle['fusion_groups']}`",
        f"- blocked reasons: `{bundle['blocked_reasons']}`",
        "",
        "Semantic deps describe graph dataflow and do not block isolated bundle development when torch-captured partition inputs are available.",
        "Implementation deps describe layout, fusion, or shared code dependencies. This bundle was built to keep implementation-coupled partitions together.",
        "If `ready` is false, stop and report the blocked status unless the main agent explicitly assigned this package with additional context.",
        "",
        "## Accuracy Target",
        "",
        *accuracy_lines(precision),
        "",
        "## Allowed Output Area",
        "",
        f"Write implementation artifacts only under `{paths['output_dir']}` and logs only under `{paths['log_dir']}`.",
        "",
        "## Procedure",
        "",
        "1. Inspect `bundle.json` and every file under `partitions/` for operation sequence, inputs, outputs, shapes, dtypes, and dependencies.",
        "2. Read the SWFT flow docs and implementation notes from `INPUT_MANIFEST.json` before writing DSL.",
        "3. Reuse patterns from the similar DSL implementation where applicable.",
        "4. Implement each owned partition under the partition-owned output directory listed in `OUTPUT_CONTRACT.json`.",
        "5. Use `slice_to_ub` for GM reads and `insert_to_gm` for GM writes unless you document a reason not to.",
        "6. Compile and run only this bundle's partition test cases.",
        "7. Compare actual outputs against the golden bins recorded in `golden_manifest.json`.",
        "8. Write every required output contract file, including bundle-level summary reports.",
        "",
        "## Completion Response",
        "",
        "Report only files changed, validation commands run, pass/fail status, and unresolved issues.",
        "",
    ])


def render_judge_task(bundle: dict[str, Any], manifest: dict[str, Any], precision: dict[str, Any]) -> str:
    partition_list = ", ".join(f"`{item}`" for item in bundle["partition_ids"])
    return "\n".join([
        f"# Subagent Judge Task: Review DSL Bundle `{bundle['bundle_id']}`",
        "",
        "## Role",
        "",
        "You are an independent judge agent. Do not continue implementation.",
        "",
        "## Objective",
        "",
        f"Judge whether the work package output for these partitions is correct and ready for the stage 05 main agent to integrate: {partition_list}.",
        "",
        "## Inputs",
        "",
        "Read these files from this judge package first:",
        "",
        "- `JUDGE_INPUT_MANIFEST.json`: paths to bundle metadata, expected outputs, and report destination.",
        "- `JUDGE_GUIDE.md`: general judging rules.",
        "",
        "Then inspect the work outputs listed in `bundle_output_checks`. Re-check paths on disk; any existence flags may reflect packaging time.",
        "",
        "## Precision Contract",
        "",
        *accuracy_lines(precision),
        "",
        "## Checks",
        "",
        "1. Confirm the work agent only implemented the partitions listed in this bundle.",
        "2. Confirm every partition has implementation.py, correctness_report.json, and validation_notes.md.",
        "3. Confirm bundle_impl_manifest.json and bundle_correctness_report.json summarize all owned partitions.",
        "4. Confirm correctness reports compare DSL actuals against torch-generated golden bins.",
        "5. Confirm relative error satisfies the configured partition tolerance, or that any override is explicit and justified.",
        "6. Confirm no shared target_dsl files, upstream stage outputs, golden files, or other bundle output directories were modified by this bundle work.",
        "7. Confirm blocked bundles were not passed unless the work output documents the missing dependency resolution.",
        "",
        "## Required Report",
        "",
        f"Write exactly one report to `{manifest['judge_report']}` with this JSON shape:",
        "",
        "```json",
        "{",
        f'  "stage": "{manifest["stage"]}",',
        f'  "bundle_id": "{bundle["bundle_id"]}",',
        '  "passed": false,',
        '  "reviewed_files": [],',
        '  "checked_items": [],',
        '  "findings": [],',
        '  "required_fixes": []',
        "}",
        "```",
        "",
        "Set `passed` to true only when this bundle can be safely consumed by the stage 05 main agent.",
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
    bundle_judge_root = stage_base / "judge" / "subagents"
    if args.clean and bundle_judge_root.exists():
        shutil.rmtree(bundle_judge_root)
    bundle_judge_root.mkdir(parents=True, exist_ok=True)

    partition_plan_ref = subagent_cfg.get("partition_plan", "stage:02_partition/output/partition_plan.json")
    partition_plan_path = Path(resolve_input_ref(config, partition_plan_ref, workspace)["path"])
    if not partition_plan_path.exists():
        raise SystemExit(f"partition plan does not exist: {partition_plan_path}")
    partition_plan = load_json(partition_plan_path)
    partitions = extract_partitions(partition_plan)
    dep_by_id = {item["id"]: dependency_info(item) for item in partitions}

    requested = None
    if args.partitions:
        requested = {sanitize_id(item.strip(), item.strip()) for item in args.partitions.split(",") if item.strip()}
        partitions = [item for item in partitions if item["id"] in requested]
        missing = sorted(requested - {item["id"] for item in partitions})
        if missing:
            raise SystemExit(f"requested partitions not found: {', '.join(missing)}")

    bundles = make_bundles(
        partitions,
        dep_by_id,
        max(1, args.max_bundle_partitions),
        merge_small_bundles=not args.no_small_bundles,
    )
    all_bundles = bundles
    if args.ready_only:
        bundles = [bundle for bundle in bundles if bundle["ready"]]

    shared_inputs = []
    for ref in stage.get("input_refs", []):
        shared_inputs.append(resolve_input_ref(config, ref, workspace))

    created = []
    for bundle in bundles:
        bundle_id = bundle["bundle_id"]
        base = package_dir / bundle_id
        if base.exists():
            shutil.rmtree(base)
        work_package = base / "work_package"
        judge_package = base / "judge_package"
        output_dir = stage_base / "output" / "bundles" / bundle_id
        log_dir = stage_base / "logs" / "bundles" / bundle_id
        judge_report_dir = bundle_judge_root / bundle_id
        if judge_report_dir.exists():
            shutil.rmtree(judge_report_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        judge_report_dir.mkdir(parents=True, exist_ok=True)
        (work_package / "partitions").mkdir(parents=True, exist_ok=True)
        judge_package.mkdir(parents=True, exist_ok=True)

        partition_output_dirs = {}
        partition_required_outputs = []
        partition_specs = []
        for partition in bundle["partitions"]:
            partition_id = partition["id"]
            part_output_dir = output_dir / "partitions" / partition_id
            part_output_dir.mkdir(parents=True, exist_ok=True)
            partition_output_dirs[partition_id] = str(part_output_dir.resolve())
            partition_required_outputs.extend([
                f"{partition_output_dirs[partition_id]}/implementation.py",
                f"{partition_output_dirs[partition_id]}/correctness_report.json",
                f"{partition_output_dirs[partition_id]}/validation_notes.md",
            ])
            partition_payload = {
                "id": partition_id,
                "index": partition["index"],
                "dependency_info": dep_by_id[partition_id],
                "bundle_id": bundle_id,
                "spec": partition["spec"],
            }
            partition_specs.append(partition_payload)
            save_json(work_package / "partitions" / f"{partition_id}.json", partition_payload)

        paths = {
            "output_dir": str(output_dir.resolve()),
            "log_dir": str(log_dir.resolve()),
            "judge_report": str((judge_report_dir / "JUDGE_REPORT.json").resolve()),
            "partition_output_dirs": partition_output_dirs,
        }
        bundle_payload = {
            "bundle_id": bundle_id,
            "index": bundle["index"],
            "partition_ids": bundle["partition_ids"],
            "bundle_reason": bundle["bundle_reason"],
            "ready": bundle["ready"],
            "blocked_reasons": bundle["blocked_reasons"],
            "semantic_deps": bundle["semantic_deps"],
            "implementation_deps": bundle["implementation_deps"],
            "fusion_groups": bundle["fusion_groups"],
            "small_elementwise_partitions": bundle["small_elementwise_partitions"],
            "paths": paths,
        }
        save_json(work_package / "bundle.json", bundle_payload)
        input_manifest = {
            "stage": args.stage,
            "bundle_id": bundle_id,
            "partition_ids": bundle["partition_ids"],
            "created_at": now_iso(),
            "workspace": str(workspace),
            "precision": config.get("precision", {}),
            "paths": paths,
            "bundle": bundle_payload,
            "partitions": partition_specs,
            "shared_inputs": shared_inputs,
        }
        output_contract = {
            "stage": args.stage,
            "bundle_id": bundle_id,
            "partition_ids": bundle["partition_ids"],
            "required_outputs": [
                f"{paths['output_dir']}/bundle_impl_manifest.json",
                f"{paths['output_dir']}/bundle_correctness_report.json",
                f"{paths['output_dir']}/validation_notes.md",
                *partition_required_outputs,
            ],
            "allowed_output_dir": paths["output_dir"],
            "allowed_log_dir": paths["log_dir"],
        }
        save_json(work_package / "INPUT_MANIFEST.json", input_manifest)
        save_json(work_package / "OUTPUT_CONTRACT.json", output_contract)
        (work_package / "AGENT_TASK.md").write_text(
            render_work_task(bundle, paths, config.get("precision", {})),
            encoding="utf-8",
        )

        bundle_output_checks = []
        for path_str in output_contract["required_outputs"]:
            path = Path(path_str)
            bundle_output_checks.append({
                "path": path_str,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            })
        judge_manifest = {
            "stage": args.stage,
            "bundle_id": bundle_id,
            "partition_ids": bundle["partition_ids"],
            "created_at": now_iso(),
            "workspace": str(workspace),
            "precision": config.get("precision", {}),
            "bundle": bundle_payload,
            "work_package": str(work_package.resolve()),
            "work_input_manifest": str((work_package / "INPUT_MANIFEST.json").resolve()),
            "work_output_contract": str((work_package / "OUTPUT_CONTRACT.json").resolve()),
            "bundle_output_checks": bundle_output_checks,
            "judge_report": paths["judge_report"],
        }
        save_json(judge_package / "JUDGE_INPUT_MANIFEST.json", judge_manifest)
        (judge_package / "JUDGE_TASK.md").write_text(
            render_judge_task(bundle, judge_manifest, config.get("precision", {})),
            encoding="utf-8",
        )
        guide = Path(__file__).resolve().parents[1] / "docs" / "JUDGE_GUIDE-CH.md"
        if guide.exists():
            shutil.copy2(guide, judge_package / "JUDGE_GUIDE.md")

        created.append({
            "bundle_id": bundle_id,
            "partition_ids": bundle["partition_ids"],
            "bundle_reason": bundle["bundle_reason"],
            "ready": bundle["ready"],
            "blocked_reasons": bundle["blocked_reasons"],
            "semantic_deps": bundle["semantic_deps"],
            "implementation_deps": bundle["implementation_deps"],
            "fusion_groups": bundle["fusion_groups"],
            "package_path": str(base.resolve()),
            "work_package_path": str(work_package.resolve()),
            "judge_package_path": str(judge_package.resolve()),
            "output_dir": paths["output_dir"],
            "log_dir": paths["log_dir"],
            "judge_report": paths["judge_report"],
        })

    blocked = [
        {
            "bundle_id": bundle["bundle_id"],
            "partition_ids": bundle["partition_ids"],
            "index": bundle["index"],
            "bundle_reason": bundle["bundle_reason"],
            "blocked_reasons": bundle["blocked_reasons"],
            "implementation_deps": bundle["implementation_deps"],
            "fusion_groups": bundle["fusion_groups"],
        }
        for bundle in all_bundles
        if not bundle["ready"]
    ]

    manifest = {
        "stage": args.stage,
        "created_at": now_iso(),
        "package_dir": str(package_dir.resolve()),
        "launch_mode": "manual_sessions",
        "ready_only": args.ready_only,
        "bundle_policy": {
            "fusion_group": "partitions with the same fusion_group are packaged together",
            "implementation_deps": "partitions with implementation_deps that point to other known partitions are packaged together",
            "can_implement_independently_false": "non-independent partitions are bundled with semantic deps when possible; unresolved singletons stay blocked",
            "small_adjacent_elementwise": "adjacent small independent elementwise partitions are merged up to max_bundle_partitions",
            "max_bundle_partitions": max(1, args.max_bundle_partitions),
            "small_bundle_merge_enabled": not args.no_small_bundles,
        },
        "bundle_count": len(created),
        "partition_count": sum(len(item["partition_ids"]) for item in created),
        "ready_bundle_count": len([item for item in created if item["ready"]]),
        "blocked_bundle_count": len(blocked),
        "bundles": created,
        "ready_bundles": [item for item in created if item["ready"]],
        "blocked_bundles": blocked,
        "ready_partitions": [partition_id for item in created if item["ready"] for partition_id in item["partition_ids"]],
        "blocked_partitions": [partition_id for item in blocked for partition_id in item["partition_ids"]],
        "manual_launch_instructions": [
            "Open one fresh work agent session per work_package_path that you want to run.",
            "Give the work agent only the work_package_path and ask it to follow AGENT_TASK.md.",
            "After the work agent finishes and the stage 05 main agent does any mechanical review, open a fresh judge agent session with the matching judge_package_path.",
            "Give the judge agent only the judge_package_path and ask it to follow JUDGE_TASK.md.",
            "Do not let work agents edit shared target_dsl files or other bundle output directories.",
            "After work and judge agents finish, the stage 05 main agent reviews accepted bundle outputs and writes the aggregate manifests."
        ],
    }
    manifest_path = stage_base / "output" / "subagent_task_manifest.json"
    save_json(manifest_path, manifest)
    print(f"created {len(created)} subagent bundle package(s): {package_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
