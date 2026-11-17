# Data preparation and separate release

## What the loaders actually need

Manifests have a top-level `motions` mapping, keyed by skill. Motion and embedding
paths are resolved **relative to the manifest**, not a developer's home directory.
Each entry includes `file`, `clip_file`, nonnegative `weight`, caption strings,
`emotion/style`, `obj_range`, and optionally contact and reference-initialization ranges.
Copy the original manifest when reconstructing a dataset: these fields drive controller initialization and contacts.

`ref_motion_smpl.npy` in the processed directories is a serialized PoseLib
`SkeletonMotion`, not an arbitrary array of SMPL parameters. It includes the skeleton,
local rotations, root translation, FPS, and cached kinematics. The released controller
uses the 15-body AMP humanoid with 28 DOFs and 125 motion features per frame.
`clip_embedding.npy` contains one or more 512-dimensional text features matching the captions.
Only load trusted motion/checkpoint files: legacy NumPy serialization uses pickle.

Processed motions and cached text features are sufficient for encoder training;
SMPL/SMPL-X model parameter files are not part of this package.
If providing the processed data, keep preprocessing dependencies and ownership documented
without bundling proprietary body-model files.

## Local inventory, November 17, 2026

The inventory was built from the adjacent `sims` experiment tree, not inferred from directory names.
Counts are manifest entries, **not deduplicated raw captures or unique actors**.

| Processed dataset | Skill entries | Captions |
| --- | --- | ---: |
| SAMP | getup 35; liedown 68; loco 138; sitdown 196; standup 97 | 2,025 |
| 100STYLE | loco 580 | 2,960 |
| ViconStyle (`vicon`) | carrywith 112; getup 124; idle 116; liedown 124; pickup 112; putdown 112 | 2,604 |
| COUCH | sitdown 424; standup 282 | 1,966 |

The four manifests and their transitive motion/embedding/object-trajectory references total **5,336 files,
455.4 MiB**, with no missing referenced files at inventory time.
The full SHA-256 inventory is a local artifact at `artifacts/motion_inventory.json`,
excluded from Git. Directory sizes are much larger because they include material
that the released loaders do not consume.

ViconStyle is the first candidate to review for a separate authored-data release.
Its manifest also contains skills excluded from this code release; the presence of
those clips does not imply this code supports carrying/pickup/putdown.
Do not rename the `carrywith` entries to another skill without checking semantics.

## Build an inventory

```bash
python scripts/motion_assets.py inventory --root /path/to/original/experiment \
  --manifest motion_data/vicon/dataset_vicon.yaml --sha256 \
  --output artifacts/vicon_inventory.json
```

This follows only the specified manifests and their motion/embedding references,
plus optional `object_file`/`obj_file` trajectories so the entire archived manifest remains usable.
It rejects references outside the data root and records missing assets without
opening pickled motion files. It does not establish redistribution rights.

## Package reviewed data

Start with `docs/redistribution_review.example.json`. Record the dataset-specific
release terms, reviewer, and an explicit approval only after the ownership,
participant permissions, and processing-derived restrictions have been checked.
Third-party datasets are separate review items; do not bundle SAMP, COUCH,
100STYLE, AMASS, CIRCLE, 3D-FRONT, or body-model files simply because they are local.
No redistribution approval or license selection has been made by this preparation step.

```bash
python scripts/motion_assets.py pack --root /path/to/original/experiment \
  --inventory artifacts/vicon_inventory.json --review artifacts/vicon_review.json \
  --output artifacts/viconstyle_processed.tar.gz
```

Packaging requires explicit approval for every manifest, verifies the complete
file set and hashes against the current sources, refuses existing output archives,
and preserves relative directory structure. It adds the inventory and review record
inside the archive and prints an archive SHA-256. Nothing is uploaded.
Extract a reviewed archive at the project root to recreate `motion_data/...`.

## Scenes and weights

For scene-free asset preparation, `scripts/make_toy_scene.py` generates toy furniture
matching the loader's metadata/mesh/point-cloud interfaces. It is suitable only for
pipeline checks, not reported metrics or a claim of benchmark generalization.

Original scenes require `scene_data/obj_info_3dfront.json` for training and
`scene_data/3dfront/model_info.json`, layout JSON, and per-object meshes, point clouds,
and affordance labels for the FSM. These are external assets, not included in Git.

Encoder and controller checkpoint paths are in the configuration files. Keep
checkpoint training provenance, associated dataset permissions, software versions,
and checksums with any later weight release. The existing local weights have not
been represented as a reviewed public model release.
