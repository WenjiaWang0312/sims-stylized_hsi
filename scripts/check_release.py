import argparse
import ast
import importlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def local_module_exists(name):
    relative = ROOT.joinpath(*name.split("."))
    return relative.with_suffix(".py").is_file() or relative.is_dir()


def main():
    parser = argparse.ArgumentParser(description="Check release syntax, local imports, assets, and optional runtime dependencies.")
    parser.add_argument("--assets", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    errors = []
    notices = []
    for path in (ROOT / "sims").rglob("*.py"):
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        if "strictly prohibited" in text:
            notices.append(str(path.relative_to(ROOT)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
                if node.module in ("sims.learning", "sims.utils", "sims.env.tasks"):
                    names += [node.module + "." + alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name.startswith("sims.") and not local_module_exists(name):
                    errors.append("{}: missing local module {}".format(path.relative_to(ROOT), name))
    for path in (ROOT / "sims/data/cfg").rglob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        if not args.assets or "env" not in config:
            continue
        env = config["env"]
        references = list(env.get("motion_file", []))
        obj_file = env.get("obj", {}).get("objFile", [])
        references.extend([obj_file] if isinstance(obj_file, str) else obj_file)
        references.extend(value for key, value in env.get("condition", {}).items() if key.startswith("motion_enc_ckpt"))
        for reference in references:
            if not (ROOT / reference).is_file():
                errors.append("{}: missing {}".format(path.name, reference))
    if args.assets:
        config = yaml.safe_load((ROOT / "sims/data/cfg/train/rlg/sims_humanoid.yaml").read_text())
        for skill, filename in config["params"]["config"]["checkpoints"].items():
            if not (ROOT / filename).is_file():
                errors.append("FSM {}: missing {}".format(skill, filename))
    if args.runtime:
        sys.path.insert(0, str(ROOT))
        for name in ["isaacgym", "torch", "rl_games", "clip", "open3d", "sims.run"]:
            try:
                importlib.import_module(name)
            except Exception as error:
                errors.append("{}: {}: {}".format(name, type(error).__name__, error))
        try:
            from rl_games.torch_runner import Runner
            if not hasattr(Runner(), "model_builder"):
                errors.append("rl_games: incompatible Runner API; install rl-games==1.1.4.")
            from sims.utils.parse_task import TASKS
            for task, module in TASKS.items():
                getattr(importlib.import_module("sims.env.tasks." + module), task)
        except Exception as error:
            errors.append("tasks: {}: {}".format(type(error).__name__, error))
    print(json.dumps({"errors": sorted(set(errors)), "license_review_required": sorted(notices)}, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
