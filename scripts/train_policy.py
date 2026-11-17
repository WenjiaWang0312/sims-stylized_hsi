import argparse
from pathlib import Path
import subprocess
import sys


SKILLS = {
    "loco": ("HumanoidSIMSLoco", "loco", "amp_humanoid_task"),
    "sitdown": ("HumanoidSIMSSit", "sit", "amp_humanoid_task_disc"),
    "liedown": ("HumanoidSIMSLie", "lie", "amp_humanoid_task_disc"),
    "standup": ("HumanoidSIMSStandup", "standup", "amp_humanoid_task"),
    "getup": ("HumanoidSIMSGetup", "getup", "amp_humanoid_task"),
    "idle": ("HumanoidSIMSIdle", "idle", "amp_humanoid_task_disc"),
    "touch": ("HumanoidSIMSTouch", "touch", "amp_humanoid_task_disc"),
}


def main():
    parser = argparse.ArgumentParser(description="Train or inspect one SIMS policy; select GPU using CUDA_VISIBLE_DEVICES.")
    parser.add_argument("skill", choices=SKILLS)
    parser.add_argument("--name", default="")
    parser.add_argument("--checkpoint")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-row", type=int, default=8)
    parser.add_argument("--num-col", type=int, default=8)
    args, extra = parser.parse_known_args()
    if min(args.num_row, args.num_col) < 1:
        parser.error("num-row and num-col must be positive.")
    if args.test and not args.checkpoint:
        parser.error("--test requires --checkpoint.")
    root = Path(__file__).resolve().parents[1]
    task, environment, training = SKILLS[args.skill]
    command = [
        sys.executable, "-m", "sims.run", "--task", task,
        "--cfg_env", "sims/data/cfg/humanoid_{}.yaml".format(environment),
        "--cfg_train", "sims/data/cfg/train/rlg/{}.yaml".format(training),
        "--num_row", str(args.num_row), "--num_col", str(args.num_col),
        "--exp_name", args.name or args.skill,
    ]
    if args.checkpoint:
        command += ["--checkpoint", str(Path(args.checkpoint).resolve())]
        if not args.test:
            command += ["--resume", "1"]
    if args.test:
        command += ["--test"]
    if args.headless or not args.test:
        command += ["--headless"]
    raise SystemExit(subprocess.call(command + extra, cwd=str(root)))


if __name__ == "__main__":
    main()
