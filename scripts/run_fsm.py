import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Execute a manually authored text-conditioned plan; no LLM needed.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--headless", action="store_true")
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, "-m", "sims.run", "--task", "Humanoid3DFrontTest",
        "--cfg_env", "sims/data/cfg/humanoid_sims_3dfront.yaml",
        "--cfg_train", "sims/data/cfg/train/rlg/sims_humanoid.yaml",
        "--num_row", "1", "--num_col", "1", "--test",
        "--plan_file", str(Path(args.plan).resolve()),
        "--record_name", Path(args.plan).stem,
    ]
    if args.headless:
        command.append("--headless")
    raise SystemExit(subprocess.call(command + extra, cwd=str(root)))


if __name__ == "__main__":
    main()
