#!/usr/bin/env python3
"""Print a compact pipeline state summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from harness_common import load_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="work")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = load_state(Path(args.workspace).resolve())
    print(f"workspace: {state['workspace']}")
    print(f"current_stage: {state['current_stage']}")
    for stage_id, info in state["stages"].items():
        judge = info.get("judge_passed")
        judge_text = "unknown" if judge is None else str(judge).lower()
        print(f"{stage_id}: {info['status']} judge={judge_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
