from importlib import import_module

import numpy as np


TASKS = {
    "HumanoidSIMSSit": "humanoid_sims_sit",
    "HumanoidSIMSLie": "humanoid_sims_lie",
    "HumanoidSIMSLoco": "humanoid_sims_loco",
    "HumanoidSIMSIdle": "humanoid_sims_idle",
    "HumanoidSIMSTouch": "humanoid_sims_touch",
    "HumanoidSIMSStandup": "humanoid_sims_standup",
    "HumanoidSIMSGetup": "humanoid_sims_getup",
    "Humanoid3DFrontTest": "humanoid_3dfront_test",
}


def parse_task(args, cfg, cfg_train, sim_params):
    if args.task not in TASKS:
        raise ValueError("Unsupported task: {}. Choose from {}".format(args.task, sorted(TASKS)))
    if args.task == "Humanoid3DFrontTest" and cfg["env"]["numEnvs"] != 1:
        raise ValueError("The minimal FSM demo supports one environment only.")
    from sims.env.tasks.vec_task_wrappers import VecTaskPythonWrapper

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg["env"]["seed"] = cfg["seed"]
    module = import_module("sims.env.tasks." + TASKS[args.task])
    task_class = getattr(module, args.task)
    task = task_class(
        cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine,
        device_type=args.device, device_id=args.device_id, headless=args.headless,
    )
    env = VecTaskPythonWrapper(
        task, args.rl_device, cfg_train.get("clip_observations", np.inf),
        cfg_train.get("clip_actions", 1.0),
    )
    return task, env
