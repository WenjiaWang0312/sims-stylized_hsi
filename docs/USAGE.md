# SIMS usage guide

## 🧩 What's included

- Train the 125-feature, 300-frame motion autoencoder and its 512-to-64-dimensional CLIP text projection.
- Train individual text-conditioned AMP controllers for locomotion, sitting, lying, and idle.
- Train the supporting standup, getup, and touch controllers.
- Execute manually authored captions, object targets, and contacts with the original task-completion FSM and policy bank.
- Inventory and package processed motion data separately from code.

LLM/RAG inference, automatic planning, carrying, paper evaluation suites, and rendering pipelines are outside this release.
The FSM demo supports **one environment**; policy training supports parallel environments.
Standup, getup, and touch are supporting controllers. Their training configurations disable conditional discrimination.

## Installation

Use Python 3.8 and an NVIDIA GPU for Isaac Gym policy training and FSM execution.
Install the separately obtained Isaac Gym package before running the controller.
Do not install a second, unrelated package named `isaacgym` from a package index.

```bash
git clone git@github.com:WenjiaWang0312/sims-stylized_hsi.git
cd sims-stylized_hsi

conda create -n sims python=3.8
conda activate sims
# Install a compatible PyTorch / torchvision pair and Isaac Gym in this environment.
pip install -r requirements.txt
pip install -e .
python scripts/check_release.py
python scripts/check_release.py --runtime
```

The control code requires **rl-games==1.1.4**, not 1.6.1: it uses the older Runner/model-builder API.
See the [release notes](RELEASE_STATUS.md) for NVIDIA notices and third-party licensing details.

## 📦 Data and assets

Download the [processed motion data from Hugging Face](https://huggingface.co/datasets/WenjiaWang/SIMS_data_release) (389 MiB)
and extract the archive in the repository root. Scene assets and model weights are separate from this download.
See [data preparation](DATA.md) for the expected contents and directory structure.
Run all commands from the `sims-stylized_hsi` repository root.

```text
motion_data/{100style,samp,vicon,couch,...}/dataset_*.yaml
motion_data/<dataset>/motions/.../{ref_motion_smpl.npy,clip_embedding.npy}
scene_data/obj_info_3dfront.json
scene_data/3dfront/{model_info.json,obj_models/,layouts/}
output/motionclip_{hsi,loco}/latest.pth
```

For a small, asset-independent **scene** smoke test, generate procedural furniture:

```bash
python scripts/make_toy_scene.py
```

This generates the metadata, collision meshes, point clouds, and contact labels expected by the loaders.
It does not generate reference motions, train a useful policy, or replace the paper's benchmark scenes.
The generator refuses to overwrite an existing scene dataset.

## Training

### MotionCLIP

```bash
python -m sims.learning.sims_motion_enc_trainer \
  --motion_file motion_data/samp/dataset_samp.yaml motion_data/vicon/dataset_vicon.yaml motion_data/couch/dataset_couch.yaml \
  --allowed_skills sitdown liedown --log_dir output/motionclip_hsi \
  --num_epochs 100 --batch_size 256

python -m sims.learning.sims_motion_enc_trainer \
  --motion_file motion_data/100style/dataset_100style.yaml motion_data/vicon/dataset_vicon.yaml \
  --allowed_skills loco idle --log_dir output/motionclip_loco \
  --num_epochs 100 --batch_size 256
```

Cached CLIP embeddings are required by default; missing embeddings produce an error instead of zero supervision.
Use `--extract_from_text` to compute them online with frozen CLIP ViT-B/32, which requires its weights.
Set `SIMS_CLIP_CACHE` to a model cache directory when pre-provisioning those weights.
Use `--device cpu` for a small encoder smoke test only.
Every completed run writes `latest.pth`, including when the final epoch is not a save-interval multiple.
These are plain state dictionaries compatible with the policy loader; training options are saved in `args.json`.

### Controllers

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_policy.py loco --name loco
CUDA_VISIBLE_DEVICES=0 python scripts/train_policy.py sitdown --name sit
CUDA_VISIBLE_DEVICES=0 python scripts/train_policy.py standup --name standup
```

Other skill names: `liedown`, `getup`, `idle`, `touch`.
The default is 8 x 8 environments; adjust `--num-row` and `--num-col` to available GPU memory.
The runner adjusts minibatches to the rollout size for smaller smoke runs.
Touch additionally requires the configured CIRCLE/AMASS-derived processed motions.
Policy training also uses cached text features by default; set
`env.condition.extractTextFeatures: true` for online extraction.

Resume with `--checkpoint /path/to/model.pth`. Inspect a trained skill with
`--test --checkpoint /path/to/model.pth --num-row 1 --num-col 1`.
Forwarded runner arguments, such as `--max_iterations`, are supported.

## Demo

1. Set the trained policy paths in `sims/data/cfg/train/rlg/sims_humanoid.yaml`.
2. Set encoder/scene paths in `sims/data/cfg/humanoid_sims_3dfront.yaml`.
3. Edit `examples/walk_sit_stand.yaml` with the object names from your scene layout.

```bash
python scripts/run_fsm.py --plan examples/walk_sit_stand.yaml --headless
```

Only policies referenced by the plan are loaded. This example needs loco, sitdown, and standup, not all seven policies.
The existing task-success thresholds and finish counters advance the FSM; changing the caption conditions the policy,
but does not automatically infer a new skill, object, or plan.
The final state is held until the episode ends. Recorded arrays are written to `render/<plan-name>.npz`.
To write your own sequence, see the [plan format and contact conventions](PLANS.md).

## 🔎 Check your setup

`python scripts/check_release.py --assets` reports missing configured manifests, scene metadata, and checkpoints;
it is a whole-release inventory, not a guarantee that all referenced meshes or every motion are valid.
Use the motion inventory tool for transitive motion dependencies.
See [validation status](RELEASE_STATUS.md) for checks performed, known limitations, and outstanding release requirements.
The smoke tests do not establish full training convergence or reproduce the paper's reported metrics.
