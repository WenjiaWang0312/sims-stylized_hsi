from pathlib import Path

import yaml


SKILLS = ("loco", "sitdown", "liedown", "standup", "getup", "idle", "touch")


def load_plan_file(filename):
    with Path(filename).open() as stream:
        plan = yaml.safe_load(stream)
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a YAML mapping.")
    scenes = plan.get("sceneJson")
    if isinstance(scenes, str):
        scenes = [scenes]
    if not isinstance(scenes, list) or not scenes or not all(isinstance(scene, str) and scene for scene in scenes):
        raise ValueError("sceneJson must be a scene layout path or a nonempty list of paths.")
    tasks = plan.get("highLevelOrder")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("highLevelOrder must contain at least one task.")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or task.get("skill") not in SKILLS:
            raise ValueError("Task {} must use one of {}.".format(index, SKILLS))
        if not isinstance(task.get("object"), str) or not task["object"].strip():
            raise ValueError("Task {} requires a nonempty object name.".format(index))
        if not isinstance(task.get("caption"), str):
            raise ValueError("Task {} requires a caption string.".format(index))
        contacts = task.get("object_contact")
        if task["skill"] == "touch":
            if contacts not in ("lhand", "rhand"):
                raise ValueError("Touch tasks require object_contact: lhand or rhand.")
        elif contacts != "none" and (
            not isinstance(contacts, list) or not all(isinstance(contact, str) and len(contact.split("-")) == 3 for contact in contacts)
        ):
            raise ValueError("Task {} requires object_contact: none or a list of joint-part-contact triples.".format(index))
    return plan
