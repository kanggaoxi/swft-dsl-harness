#!/usr/bin/env python3
"""Advance pipeline_state.json to the next stage after a passed gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from harness_common import DEFAULT_CONFIG, load_config, load_state, next_stage_id, now_iso, save_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="work")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    state = load_state(workspace)
    config_path = Path(args.config).resolve() if args.config else Path(state.get("pipeline_config", DEFAULT_CONFIG)).resolve()
    config = load_config(config_path)
    current = state["current_stage"]
    if state["stages"][current]["status"] != "passed":
        raise SystemExit(f"current stage {current} is not passed")

    nxt = next_stage_id(config, current)
    if not nxt:
        state["updated_at"] = now_iso()
        state["pipeline_status"] = "complete"
        save_state(workspace, state)
        print(f"pipeline complete at stage {current}")
        return 0

    state["current_stage"] = nxt
    if state["stages"][nxt]["status"] == "pending":
        state["stages"][nxt]["status"] = "ready"
    state["updated_at"] = now_iso()
    save_state(workspace, state)
    print(f"advanced: {current} -> {nxt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
