#!/usr/bin/env python3
"""Initialize a SWFT DSL harness workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

from harness_common import DEFAULT_CONFIG, create_stage_dirs, init_state, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="harness/work")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    config_path = Path(args.config).resolve()
    state_file = workspace / "pipeline_state.json"
    if state_file.exists() and not args.force:
        raise SystemExit(f"state already exists: {state_file}; use --force to reinitialize")

    config = load_config(config_path)
    workspace.mkdir(parents=True, exist_ok=True)
    create_stage_dirs(workspace, config)
    save_json(workspace / "pipeline_config.snapshot.json", config)
    save_json(state_file, init_state(workspace, config_path, config))

    readme = workspace / "README.md"
    if not readme.exists() or args.force:
        readme.write_text(
            "# Harness Workspace\n\n"
            "Place user-provided inputs here:\n\n"
            "- `shared/model/model.py`\n"
            "- `shared/model/weights.pth`\n"
            "- `shared/model/input_spec.json`\n"
            "- `shared/similar_dsl/similar_model_dsl.py`\n\n"
            "Generated stage outputs live under `stages/<stage_id>/output/`.\n",
            encoding="utf-8",
        )

    print(f"initialized workspace: {workspace}")
    print(f"current stage: {config['stages'][0]['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
