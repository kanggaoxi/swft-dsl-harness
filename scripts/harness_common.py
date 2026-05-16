#!/usr/bin/env python3
"""Shared helpers for the SWFT DSL harness."""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT
DEFAULT_CONFIG = HARNESS_ROOT / "configs" / "pipeline.default.json"


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_json(config_path or DEFAULT_CONFIG)


def stage_by_id(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in config["stages"]:
        if stage["id"] == stage_id:
            return stage
    raise KeyError(f"unknown stage: {stage_id}")


def stage_ids(config: dict[str, Any]) -> list[str]:
    return [stage["id"] for stage in config["stages"]]


def workspace_path(path: str | Path) -> Path:
    return Path(path).resolve()


def state_path(workspace: Path) -> Path:
    return workspace / "pipeline_state.json"


def load_state(workspace: Path) -> dict[str, Any]:
    path = state_path(workspace)
    if not path.exists():
        raise FileNotFoundError(f"pipeline state does not exist: {path}")
    return load_json(path)


def save_state(workspace: Path, state: dict[str, Any]) -> None:
    save_json(state_path(workspace), state)


def create_stage_dirs(workspace: Path, config: dict[str, Any]) -> None:
    for stage_id in stage_ids(config):
        base = workspace / "stages" / stage_id
        for subdir in ("agent_package", "judge_package", "output", "logs", "validation", "judge"):
            (base / subdir).mkdir(parents=True, exist_ok=True)
    (workspace / "shared" / "model").mkdir(parents=True, exist_ok=True)
    (workspace / "shared" / "similar_dsl").mkdir(parents=True, exist_ok=True)
    (workspace / "target_dsl").mkdir(parents=True, exist_ok=True)


def init_state(workspace: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    ids = stage_ids(config)
    return {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "pipeline_config": str(config_path.resolve()),
        "workspace": str(workspace.resolve()),
        "current_stage": ids[0],
        "stages": {
            stage_id: {
                "status": "pending",
                "attempts": 0,
                "package_path": None,
                "last_validation_report": None,
                "validated_at": None,
                "last_judge_report": None,
                "judged_at": None,
                "judge_passed": None
            }
            for stage_id in ids
        }
    }


def stage_dir(workspace: Path, stage_id: str) -> Path:
    return workspace / "stages" / stage_id


def resolve_external(config: dict[str, Any], label: str, workspace: Path) -> dict[str, Any]:
    for item in config.get("external_inputs", []):
        if item["label"] == label:
            base = HARNESS_ROOT if item.get("relative_to") == "harness" else workspace
            resolved = (base / item["path"]).resolve()
            return {**item, "resolved_path": str(resolved)}
    raise KeyError(f"unknown external input: {label}")


def resolve_input_ref(config: dict[str, Any], ref: str, workspace: Path) -> dict[str, Any]:
    if ref.startswith("external:"):
        label = ref.split(":", 1)[1]
        item = resolve_external(config, label, workspace)
        path = Path(item["resolved_path"])
        return {
            "ref": ref,
            "label": label,
            "kind": "external",
            "path": str(path),
            "exists": path.exists()
        }
    if ref.startswith("stage:"):
        payload = ref.split(":", 1)[1]
        source_stage, rel = payload.split("/", 1)
        path = stage_dir(workspace, source_stage) / rel
        return {
            "ref": ref,
            "label": f"{source_stage}:{rel}",
            "kind": "stage_output",
            "stage": source_stage,
            "path": str(path.resolve()),
            "exists": path.exists()
        }
    raise ValueError(f"unsupported input ref: {ref}")


def copy_input(path: Path, destination_dir: Path) -> str | None:
    if not path.exists():
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest = destination_dir / path.name
    if path.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(path, dest)
    else:
        shutil.copy2(path, dest)
    return str(dest.resolve())


def required_output_status(stage_base: Path, rel_path: str) -> dict[str, Any]:
    if any(ch in rel_path for ch in "*?[]"):
        matches = [str(Path(p).resolve()) for p in glob.glob(str(stage_base / rel_path))]
        return {
            "path": rel_path,
            "type": "glob",
            "exists": bool(matches),
            "matches": matches
        }
    path = stage_base / rel_path
    return {
        "path": rel_path,
        "type": "path",
        "exists": path.exists(),
        "resolved_path": str(path.resolve())
    }


def run_validation_commands(commands: list[str], cwd: Path, env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    results = []
    for command in commands:
        started = now_iso()
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        results.append({
            "command": command,
            "started_at": started,
            "finished_at": now_iso(),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "passed": proc.returncode == 0
        })
    return results


def previous_stage_id(config: dict[str, Any], stage_id: str) -> str | None:
    ids = stage_ids(config)
    idx = ids.index(stage_id)
    return ids[idx - 1] if idx > 0 else None


def next_stage_id(config: dict[str, Any], stage_id: str) -> str | None:
    ids = stage_ids(config)
    idx = ids.index(stage_id)
    return ids[idx + 1] if idx + 1 < len(ids) else None
