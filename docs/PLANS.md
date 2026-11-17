# Manually authored FSM plans

`examples/walk_sit_stand.yaml` is a schema example compatible with the procedural scene.
The plan is not a natural-language planner: users supply a skill, scene object, and
caption for each state, while the controller uses the caption for its motion condition.

The loader reads:

- `sceneJson`: one layout path, or a list of layout paths, relative to `env.obj.sceneDataRoot`.
- `roomTrans`: optional translations between multiple room layouts.
- `maxEpisodeLength`: optional episode duration in simulation control steps.
- `highLevelOrder`: an ordered, nonempty list of task mappings.
- `initLocation`: optional initial object/standpoint reference using the existing loader format.

Every task has:

- `skill`: `loco`, `sitdown`, `liedown`, `standup`, `getup`, `idle`, or `touch`.
- `object`: a name in the layout's `obj_names`. A suffix such as `chair1-2` selects a standpoint.
- `caption`: a string encoded by CLIP ViT-B/32 and the trained motion text projection.
  Empty strings are supported for the original unconditioned auxiliary skills.
- `object_contact`: `none`, or a list such as `[pelvis-seat-up, head-seat-up]`.
  For `touch`, this field is instead a single joint abbreviation: `lhand` or `rhand`.

The procedural scene affordance is named `seat`. Joint abbreviations follow
`HumanoidSIMS.joint_abbr`; contact suffixes are `up`, `none`, or `nocontact`.
For sit/lie, the task adds its default support contacts unless explicitly disabled.
The plan validator checks structure before scene loading; actual object names,
mesh paths, and affordances still need to match your scene assets.

Policy dispatch is implemented by `AMPMultiPlayerContinuous`, using
`Humanoid3DFrontTest._fetch_every_env_task()`. The environment's existing task
thresholds and finish counters advance `_pointer`. This is the actual
physics task FSM, not a replacement timer-only sequencer.
The last state remains active until episode termination.
Only one simulation environment is supported by this FSM demo because the
original layout/contact code assumes scalar plan-state selection.
