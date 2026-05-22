#!/usr/bin/env python3
"""Create isolated DSL implementation and judge packages for graph family bundles."""

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
    parser.add_argument("--max-bundle-partitions", type=int, default=DEFAULT_MAX_BUNDLE_PARTITIONS, help="Deprecated compatibility option; family_group controls bundle size.")
    parser.add_argument("--no-small-bundles", action="store_true", help="Deprecated compatibility option; family_group controls bundling.")
    parser.add_argument("--clean", action="store_true", help="Remove existing agent package directory first.")
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
    family_group = (
        spec.get("family_group")
        or spec.get("family")
        or spec.get("group")
        or spec.get("repeat_group")
        or spec.get("similarity_group")
        or spec.get("fusion_group")
    )
    fusion_group = spec.get("fusion_group")
    repeat_group = spec.get("repeat_group")
    repeat_index = spec.get("repeat_index")
    repeat_role = spec.get("repeat_role")
    implementation_signature = spec.get("implementation_signature")
    prototype_ref = spec.get("prototype_ref")
    similarity_group = spec.get("similarity_group")
    shared_core = listify(spec.get("shared_core"))
    variant_delta = spec.get("variant_delta", {})
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
        "family_group": family_group,
        "fusion_group": fusion_group,
        "repeat_group": repeat_group,
        "repeat_index": repeat_index,
        "repeat_role": repeat_role,
        "implementation_signature": implementation_signature,
        "prototype_ref": prototype_ref,
        "similarity_group": similarity_group,
        "shared_core": shared_core,
        "variant_delta": variant_delta,
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


def family_group_key(partition: dict[str, Any], dep_info: dict[str, Any]) -> str:
    family_group = dep_info.get("family_group")
    if family_group in (None, ""):
        family_group = partition["spec"].get("family_group")
    if family_group in (None, ""):
        family_group = partition["spec"].get("family")
    if family_group in (None, ""):
        family_group = partition["spec"].get("group")
    if family_group in (None, ""):
        family_group = partition["id"]
    return sanitize_id(str(family_group), partition["id"])


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
    by_signature: dict[str, list[str]] = {}

    for partition in partitions:
        partition_id = partition["id"]
        info = dep_by_id[partition_id]
        fusion_group = info.get("fusion_group")
        if fusion_group not in (None, ""):
            by_fusion_group.setdefault(str(fusion_group), []).append(partition_id)
        signature = info.get("implementation_signature")
        similarity_group = info.get("similarity_group")
        if similarity_group not in (None, "") and signature not in (None, ""):
            by_signature.setdefault(f"similar:{similarity_group}:{signature}", []).append(partition_id)
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
    for group_ids in by_signature.values():
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
    family_groups = []
    fusion_groups = []
    repeat_groups = []
    repeat_roles = []
    implementation_signatures = []
    similarity_groups = []
    shared_cores = []
    variant_deltas = {}
    prototype_refs = []
    small_elementwise = []
    for partition in bundle_partitions:
        partition_id = partition["id"]
        info = dep_by_id[partition_id]
        semantic_deps.extend(info["semantic_deps"])
        implementation_deps.extend(info["implementation_deps"])
        if info["family_group"] not in (None, ""):
            family_groups.append(str(info["family_group"]))
        if info["fusion_group"] not in (None, ""):
            fusion_groups.append(str(info["fusion_group"]))
        if info["repeat_group"] not in (None, ""):
            repeat_groups.append(str(info["repeat_group"]))
        if info["repeat_role"] not in (None, ""):
            repeat_roles.append(str(info["repeat_role"]))
        if info["implementation_signature"] not in (None, ""):
            implementation_signatures.append(str(info["implementation_signature"]))
        if info["similarity_group"] not in (None, ""):
            similarity_groups.append(str(info["similarity_group"]))
        shared_cores.extend(info["shared_core"])
        if info["variant_delta"] not in (None, {}, []):
            variant_deltas[partition_id] = info["variant_delta"]
        if info["prototype_ref"] not in (None, ""):
            prototype_refs.append(str(info["prototype_ref"]))
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
        "family_groups": sorted(set(family_groups)),
        "fusion_groups": sorted(set(fusion_groups)),
        "repeat_groups": sorted(set(repeat_groups)),
        "repeat_roles": sorted(set(repeat_roles)),
        "implementation_signatures": sorted(set(implementation_signatures)),
        "similarity_groups": sorted(set(similarity_groups)),
        "shared_core": sorted(set(shared_cores)),
        "variant_deltas": variant_deltas,
        "prototype_refs": sorted(set(prototype_refs)),
        "small_elementwise_partitions": small_elementwise,
    }


def build_repeat_group_manifest(partitions: list[dict[str, Any]], dep_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for partition in partitions:
        repeat_group = dep_by_id[partition["id"]].get("repeat_group")
        if repeat_group not in (None, ""):
            grouped.setdefault(str(repeat_group), []).append(partition)

    manifests = []
    for group_name, group_partitions in sorted(grouped.items()):
        entries = []
        prototypes = []
        replicas = []
        for partition in sorted(group_partitions, key=lambda item: item["index"]):
            info = dep_by_id[partition["id"]]
            entry = {
                "partition_id": partition["id"],
                "repeat_index": info.get("repeat_index"),
                "repeat_role": info.get("repeat_role"),
                "implementation_signature": info.get("implementation_signature"),
                "prototype_ref": info.get("prototype_ref"),
            }
            entries.append(entry)
            if info.get("repeat_role") == "prototype":
                prototypes.append(partition["id"])
            elif info.get("repeat_role") == "replica":
                replicas.append(partition["id"])
        manifests.append({
            "repeat_group": group_name,
            "prototype_partitions": prototypes,
            "replica_partitions": replicas,
            "partitions": entries,
            "policy": "implement prototype partitions first; implement replicas by adapting the prototype with per-instance bindings and verify every replica against its own torch golden",
        })
    return manifests


def build_similarity_group_manifest(partitions: list[dict[str, Any]], dep_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for partition in partitions:
        similarity_group = dep_by_id[partition["id"]].get("similarity_group")
        if similarity_group not in (None, ""):
            grouped.setdefault(str(similarity_group), []).append(partition)

    manifests = []
    for group_name, group_partitions in sorted(grouped.items()):
        entries = []
        shared_core = []
        signatures = []
        for partition in sorted(group_partitions, key=lambda item: item["index"]):
            info = dep_by_id[partition["id"]]
            shared_core.extend(info.get("shared_core", []))
            if info.get("implementation_signature") not in (None, ""):
                signatures.append(str(info["implementation_signature"]))
            entries.append({
                "partition_id": partition["id"],
                "implementation_signature": info.get("implementation_signature"),
                "shared_core": info.get("shared_core", []),
                "variant_delta": info.get("variant_delta", {}),
            })
        manifests.append({
            "similarity_group": group_name,
            "implementation_signatures": sorted(set(signatures)),
            "shared_core": sorted(set(shared_core)),
            "variants": entries,
            "policy": "reuse the shared core where possible, implement only documented variant_delta differences, and verify every variant against its own torch golden",
        })
    return manifests


def make_bundles(
    partitions: list[dict[str, Any]],
    dep_by_id: dict[str, dict[str, Any]],
    max_bundle_partitions: int,
    merge_small_bundles: bool,
) -> list[dict[str, Any]]:
    id_set = {item["id"] for item in partitions}
    family_groups: dict[str, list[dict[str, Any]]] = {}
    for partition in partitions:
        info = dep_by_id[partition["id"]]
        family_key = family_group_key(partition, info)
        family_groups.setdefault(family_key, []).append(partition)

    raw_bundles = [sorted(group, key=lambda item: item["index"]) for group in family_groups.values()]
    raw_bundles.sort(key=lambda group: group[0]["index"])

    bundles = []
    for idx, group in enumerate(raw_bundles):
        status = bundle_status(group, dep_by_id, id_set)
        family_group_name = status["family_groups"][0] if status["family_groups"] else group[0]["id"]
        if len(group) > 1:
            reason = "family_group"
        else:
            reason = "single_partition"
        bundle_id = sanitize_id(family_group_name, f"family_{idx:03d}")
        bundles.append({
            "bundle_id": bundle_id,
            "index": idx,
            "family_group": family_group_name,
            "subgraph_order": [item["id"] for item in group],
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
    subgraph_lines = []
    for order, partition in enumerate(bundle["partitions"], start=1):
        spec = partition["spec"]
        family_group = spec.get("family_group") or spec.get("family") or spec.get("group") or bundle["family_group"]
        repeat_role = spec.get("repeat_role")
        repeat_group = spec.get("repeat_group")
        similarity_group = spec.get("similarity_group")
        role_bits = []
        if repeat_role:
            role_bits.append(f"repeat_role={repeat_role}")
        if repeat_group:
            role_bits.append(f"repeat_group={repeat_group}")
        if similarity_group:
            role_bits.append(f"similarity_group={similarity_group}")
        role_text = f" ({', '.join(role_bits)})" if role_bits else ""
        subgraph_lines.append(f"{order}. `{partition['id']}`{role_text} in family `{family_group}`")

    lines = [
        f"# Agent Work Task: Implement Family Bundle `{bundle['bundle_id']}`",
        "",
        "## Role",
        "",
        "You are the family work agent for one family bundle. Coordinate subgraph implementation one subgraph at a time.",
        "If you can start child agents, start exactly one subgraph worker agent for the current subgraph, wait for it to pass golden comparison, then start the next subgraph. Never run multiple subgraph worker agents in parallel.",
        "",
        "## Objective",
        "",
        f"Implement only these partitions in SWFT DSL and validate them against torch-generated golden bins: {partition_list}.",
        "",
        "Follow the subgraph order exactly. Finish one subgraph, verify it, then start the next one.",
        "",
        "Do not modify other bundles. Do not change the global partition plan, golden files, harness scripts, or upstream artifacts.",
        "",
        "## Family",
        "",
        f"- family group: `{bundle['family_group']}`",
        f"- bundle reason: `{bundle['bundle_reason']}`",
        f"- ready: `{bundle['ready']}`",
        f"- blocked reasons: `{bundle['blocked_reasons']}`",
        "",
        "## Subgraph Order",
        "",
        *subgraph_lines,
        "",
        "## Required Inputs",
        "",
        "Read these files from this package first:",
        "",
        "- `bundle.json`: bundle metadata and family order.",
        "- `partitions/*.json`: the only partitions you own.",
        "- `INPUT_MANIFEST.json`: shared paths for model IR, partition plan, golden manifest, cases, skeleton DSL, similar DSL, SWFT flow docs, and implementation notes.",
        "- `OUTPUT_CONTRACT.json`: files this family work agent must produce.",
        "",
        "## Skeleton And Output Ownership",
        "",
        "Use the stage 04 `target_model_dsl.py` only as a read-only skeleton reference for compile/run/file input/output structure.",
        "Do not edit stage 04 outputs. Do not edit or create shared `target_dsl/` files in this stage.",
        "For each subgraph, write a new partition-owned DSL implementation file at the `implementation.py` path required by `OUTPUT_CONTRACT.json`.",
        "Any temporary generated DSL, CCE, actual bins, and debug logs must stay under this bundle's allowed output or log directory.",
        "Stage 06 owns integrating accepted partition implementations into a full `target_model_dsl.py`.",
        "",
        "## Bundle Notes",
        "",
        f"- semantic deps: `{bundle['semantic_deps']}`",
        f"- implementation deps: `{bundle['implementation_deps']}`",
        f"- repeat groups: `{bundle['repeat_groups']}`",
        f"- repeat roles: `{bundle['repeat_roles']}`",
        f"- similarity groups: `{bundle['similarity_groups']}`",
        f"- shared core: `{bundle['shared_core']}`",
        f"- variant deltas: `{bundle['variant_deltas']}`",
        f"- prototype refs: `{bundle['prototype_refs']}`",
        f"- implementation signatures: `{bundle['implementation_signatures']}`",
        "",
        "Semantic deps describe graph dataflow and do not block isolated bundle development when torch-captured partition inputs are available.",
        "Implementation deps describe layout, fusion, or shared code dependencies. This bundle was built as one sequential family package.",
        "Repeat groups describe structurally identical instances. Implement the prototype first, then copy it to replicas and adapt only the per-instance bindings.",
        "Similarity groups describe related but not identical variants. Reuse the shared core where possible and implement only the documented variant deltas.",
        "If `ready` is false, stop and report the blocked status unless the main agent explicitly assigned this package with additional context.",
        "",
        "## Optional SWFT Source",
        "",
        "If `INPUT_MANIFEST.json` contains `swft_source`, use it when DSL semantics, generated CCE, or runtime behavior are unclear. Prefer targeted source searches over reading the whole compiler tree.",
        "",
        "## Accuracy Target",
        "",
        *accuracy_lines(precision),
        "",
        "## Allowed Output Area",
        "",
        f"Write implementation artifacts only under `{paths['output_dir']}` and logs only under `{paths['log_dir']}`.",
        "Do not write to `target_dsl/`, `stages/04_dsl_skeleton/`, golden directories, or other bundle output directories.",
        "",
        "## Procedure",
        "",
        "1. Read the bundle metadata and the subgraph order first.",
        "2. Read the SWFT flow docs, implementation notes, and the read-only stage 04 skeleton from `INPUT_MANIFEST.json` before writing DSL.",
        "3. For the current subgraph only, start one subgraph worker agent if available. Give it this work_package path and the current partition id; tell it to read only `partitions/<partition_id>.json`, `INPUT_MANIFEST.json`, and `OUTPUT_CONTRACT.json`.",
        "4. The subgraph worker must create or update only that partition's required `implementation.py`, correctness report, validation notes, and artifacts under this bundle output directory.",
        "5. The current subgraph must compile, run, and compare against its torch golden before any later subgraph starts.",
        "6. Reuse patterns from the similar DSL implementation where applicable.",
        "7. For repeat groups, implement the prototype first and then copy the implementation to replicas with per-instance bindings.",
        "8. Use `slice_to_ub` for GM reads and `insert_to_gm` for GM writes unless you document a reason not to.",
        "9. When debugging a DSL compile/runtime mismatch, inspect SWFT source only for the relevant frontend/API/lowering path and record the files consulted.",
        "10. After every subgraph in the listed order passes, write every required output contract file, including bundle-level summary reports.",
        "",
        "## Completion Response",
        "",
        "Report only files changed, validation commands run, pass/fail status, and unresolved issues.",
        "",
    ]
    return "\n".join(lines)


def render_judge_task(bundle: dict[str, Any], manifest: dict[str, Any], precision: dict[str, Any]) -> str:
    partition_list = ", ".join(f"`{item}`" for item in bundle["partition_ids"])
    return "\n".join([
        f"# Agent Judge Task: Review Family Bundle `{bundle['bundle_id']}`",
        "",
        "## Role",
        "",
        "You are an independent judge agent. Do not continue implementation.",
        "",
        "## Objective",
        "",
        f"Judge whether the work package output for these partitions is correct and ready for the stage 05 main agent to integrate: {partition_list}.",
        "",
        "Confirm that the work agent followed the declared subgraph order and did not process multiple subgraphs in parallel.",
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
        "2. Confirm the work output documents the declared subgraph order and sequential execution.",
        "3. Confirm every partition has implementation.py, correctness_report.json, and validation_notes.md.",
        "4. Confirm bundle_impl_manifest.json and bundle_correctness_report.json summarize all owned partitions.",
        "5. Confirm correctness reports compare DSL actuals against torch-generated golden bins.",
        "6. Confirm relative error satisfies the configured partition tolerance, or that any override is explicit and justified.",
        "7. Confirm no shared target_dsl files, upstream stage outputs, golden files, or other bundle output directories were modified by this bundle work.",
        "8. Confirm stage 04 target_model_dsl.py was used only as a read-only skeleton reference and was not modified.",
        "9. For repeat groups, confirm replicas are adapted from the prototype with correct per-instance weights/inputs/outputs and each replica has its own golden comparison.",
        "10. For similarity groups, confirm the shared core is reused where appropriate and each variant_delta is explicitly implemented and tested.",
        "11. Confirm any SWFT source conclusions cite specific files or code paths, not vague compiler assumptions.",
        "12. Confirm blocked bundles were not passed unless the work output documents the missing dependency resolution.",
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
    agent_cfg = stage.get("agent_packages") or stage.get("subagent_packages", {})
    if not agent_cfg.get("enabled"):
        raise SystemExit(f"stage {args.stage} does not enable agent packages")

    stage_base = stage_dir(workspace, args.stage)
    package_dir = workspace / agent_cfg.get("package_dir", f"stages/{args.stage}/agent_packages")
    if args.clean and package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    bundle_judge_root = stage_base / "judge" / "agents"
    if args.clean and bundle_judge_root.exists():
        shutil.rmtree(bundle_judge_root)
    bundle_judge_root.mkdir(parents=True, exist_ok=True)

    partition_plan_ref = agent_cfg.get("partition_plan", "stage:02_partition/output/partition_plan.json")
    partition_plan_path = Path(resolve_input_ref(config, partition_plan_ref, workspace)["path"])
    if not partition_plan_path.exists():
        raise SystemExit(f"partition plan does not exist: {partition_plan_path}")
    partition_plan = load_json(partition_plan_path)
    partitions = extract_partitions(partition_plan)
    dep_by_id = {item["id"]: dependency_info(item) for item in partitions}
    all_partitions = partitions

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

    repeat_group_manifest = build_repeat_group_manifest(all_partitions, dep_by_id)
    similarity_group_manifest = build_similarity_group_manifest(all_partitions, dep_by_id)

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
                "order": len(partition_specs) + 1,
                "family_group": bundle["family_group"],
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
            "family_group": bundle["family_group"],
            "partition_ids": bundle["partition_ids"],
            "subgraph_order": bundle["subgraph_order"],
            "bundle_reason": bundle["bundle_reason"],
            "ready": bundle["ready"],
            "blocked_reasons": bundle["blocked_reasons"],
            "family_groups": bundle["family_groups"],
            "semantic_deps": bundle["semantic_deps"],
            "implementation_deps": bundle["implementation_deps"],
            "fusion_groups": bundle["fusion_groups"],
            "repeat_groups": bundle["repeat_groups"],
            "repeat_roles": bundle["repeat_roles"],
            "implementation_signatures": bundle["implementation_signatures"],
            "similarity_groups": bundle["similarity_groups"],
            "shared_core": bundle["shared_core"],
            "variant_deltas": bundle["variant_deltas"],
            "prototype_refs": bundle["prototype_refs"],
            "small_elementwise_partitions": bundle["small_elementwise_partitions"],
            "subgraph_specs": partition_specs,
            "paths": paths,
        }
        save_json(work_package / "bundle.json", bundle_payload)
        input_manifest = {
            "stage": args.stage,
            "bundle_id": bundle_id,
            "family_group": bundle["family_group"],
            "partition_ids": bundle["partition_ids"],
            "subgraph_order": bundle["subgraph_order"],
            "created_at": now_iso(),
            "workspace": str(workspace),
            "precision": config.get("precision", {}),
            "paths": paths,
            "bundle": bundle_payload,
            "partitions": partition_specs,
            "subgraph_specs": partition_specs,
            "repeat_groups": repeat_group_manifest,
            "similarity_groups": similarity_group_manifest,
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
            "family_group": bundle["family_group"],
            "partition_ids": bundle["partition_ids"],
            "subgraph_order": bundle["subgraph_order"],
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
            "family_group": bundle["family_group"],
            "partition_ids": bundle["partition_ids"],
            "subgraph_order": bundle["subgraph_order"],
            "bundle_reason": bundle["bundle_reason"],
            "ready": bundle["ready"],
            "blocked_reasons": bundle["blocked_reasons"],
            "family_groups": bundle["family_groups"],
            "semantic_deps": bundle["semantic_deps"],
            "implementation_deps": bundle["implementation_deps"],
            "fusion_groups": bundle["fusion_groups"],
            "repeat_groups": bundle["repeat_groups"],
            "repeat_roles": bundle["repeat_roles"],
            "implementation_signatures": bundle["implementation_signatures"],
            "similarity_groups": bundle["similarity_groups"],
            "shared_core": bundle["shared_core"],
            "variant_deltas": bundle["variant_deltas"],
            "prototype_refs": bundle["prototype_refs"],
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
            "family_group": bundle["family_group"],
            "partition_ids": bundle["partition_ids"],
            "subgraph_order": bundle["subgraph_order"],
            "index": bundle["index"],
            "bundle_reason": bundle["bundle_reason"],
            "blocked_reasons": bundle["blocked_reasons"],
            "family_groups": bundle["family_groups"],
            "implementation_deps": bundle["implementation_deps"],
            "fusion_groups": bundle["fusion_groups"],
            "repeat_groups": bundle["repeat_groups"],
            "similarity_groups": bundle["similarity_groups"],
            "prototype_refs": bundle["prototype_refs"],
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
            "family_group": "partitions with the same family_group are packaged together for one human-managed work session",
            "subgraph_order": "partitions inside one family bundle are executed strictly in index order",
            "implementation_deps": "partitions with implementation_deps that point to other known partitions are packaged together only when they stay inside the same family bundle",
            "repeat_group": "partitions may declare repeat_group, repeat_index, repeat_role, implementation_signature, and prototype_ref so prototype work can be reused by replicas inside the family bundle",
            "similarity_group": "partitions may declare similarity_group, shared_core, implementation_signature, and variant_delta so related variants can share a core while staying inside the same family bundle",
        },
        "repeat_groups": repeat_group_manifest,
        "similarity_groups": similarity_group_manifest,
        "bundle_count": len(created),
        "partition_count": sum(len(item["partition_ids"]) for item in created),
        "ready_bundle_count": len([item for item in created if item["ready"]]),
        "blocked_bundle_count": len(blocked),
        "bundles": created,
        "ready_bundles": [item for item in created if item["ready"]],
        "blocked_bundles": blocked,
        "prototype_bundles": [item for item in created if "prototype" in item["repeat_roles"]],
        "replica_bundles": [item for item in created if "replica" in item["repeat_roles"]],
        "ready_partitions": [partition_id for item in created if item["ready"] for partition_id in item["partition_ids"]],
        "blocked_partitions": [partition_id for item in blocked for partition_id in item["partition_ids"]],
        "manual_launch_instructions": [
            "Open one fresh work agent session per ready_bundles[*].work_package_path.",
            "Give each work agent only its own work_package_path and ask it to follow AGENT_TASK.md.",
            "Inside one family bundle, the family work agent may start one subgraph worker agent for the current subgraph, but must wait for it to pass before starting the next one.",
            "After the work agent finishes, open a fresh judge agent session with the matching judge_package_path.",
            "Give the judge agent only the judge_package_path and ask it to follow JUDGE_TASK.md.",
            "Do not let work agents edit shared target_dsl files or other bundle output directories.",
            "After work and judge agents finish, the stage 05 main agent reviews accepted family bundle outputs and writes the aggregate manifests."
        ],
    }
    manifest_path = stage_base / "output" / "agent_task_manifest.json"
    save_json(manifest_path, manifest)
    print(f"created {len(created)} family agent package(s): {package_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
