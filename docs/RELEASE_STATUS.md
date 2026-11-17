# Validation and release status

Last validated November 17, 2026.

Repository: `git@github.com:WenjiaWang0312/sims-stylized_hsi.git`.
This document records local validation, not proof of paper-level reproduction.
Motion data, scene assets, and model weights are released separately from GitHub.

## Sources and bounded changes

- Primary source: the adjacent `sims-stylized_hsi/sims` migration draft, whose contents
  came from the later `sims` experiment tree.
- The missing polygon helper was reconstructed against the older `sims_iccv` implementation.
- Only AMP training, its Python simulation tasks, MotionCLIP, PoseLib dependencies,
  and the 3D-FRONT task/FSM path are retained. No LLM client is included.
- Imports use the `sims` package; the task registry loads supported modules lazily
  instead of importing missing carry/HRL/transformer experiments or using `eval`.
- Encoder data indexing now uses scalar motion indices, required CLI inputs are
  checked, checkpoints are always saved at the final epoch, and cached text features
  cannot silently become zero vectors.
- The standup launcher now selects the standup class. The loco manifest list includes
  the idle references required by its configured initialization skills.
- The FSM loads only policies present in the plan and does not need a carry encoder.
- Procedural scenes are available for smoke checks without distributing captured scene assets.

## Validation performed

- Python syntax and in-tree module checks.
- All eight retained task classes and AMP/MotionCLIP modules imported in the local Python 3.8 environment.
- A real processed motion was loaded through PoseLib and converted to finite
  `(125, 300)` motion features, a 300-frame mask, and 512-dimensional text features.
- One complete CPU encoder training epoch on one real motion, including backward,
  optimizer update, TensorBoard/log output, and a final `latest.pth` despite a
  non-divisible save interval. This is a smoke test, not a useful trained model.
- GPU controller training was exercised using one chair-compatible SAMP sitting
  reference, one locomotion reference, procedural furniture, and a smoke-trained
  encoder. A bounded run with `--max_iterations 1` completed optimizer updates
  and wrote `Humanoid.pth`. The inherited loop stops when its counter exceeds the
  configured limit, so this smoke run performed two rollout/update cycles.
- Three existing policies (loco, sitdown, standup) and the two original encoders
  were loaded with real CLIP ViT-B/32 text encoding. A 24-frame headless procedural-scene
  run produced finite body states and visited task IDs 1, 2, and 5. Success thresholds
  were deliberately relaxed for this switching check; it does not establish task
  success or motion quality under the paper's thresholds.
- Plan validation and data packaging were exercised with synthetic fixtures,
  including rejection of unreviewed data, changed files, and paths escaping the data root.
- Four dataset manifests and all 5,336 referenced files were inventoried and SHA-256 hashed.
- Nine original encoder/controller checkpoints were located and SHA-256 hashed
  in `artifacts/checkpoint_inventory.json`, without modifying or uploading them.

The installed `gym` environment has rl-games 1.6.1, which imports but fails at model
registration. The requirements now pin 1.1.4, matching the adjacent ASE requirements.
Do not treat import-only checks as evidence that 1.6.1 can train these policies.
For physics checks, the existing 1.1.4 installation from the adjacent ASE environment
was selected through a temporary `PYTHONPATH`, without downgrading the user's gym environment.
The tested stack was Python 3.8.20, PyTorch 2.4.1 + CUDA 12.1,
torchvision 0.19.1, NumPy 1.23.0, Open3D 0.18.0, and an RTX 4090.
No paper success rates or full training convergence are claimed.

## Publication gates

1. Confirm the intended license for the authored code. Preserve all third-party notices.
   Several retained NVIDIA files contain restrictive notices; `check_release.py`
   lists them under `license_review_required`. Their upstream provenance/release
   permission must be established or appropriately replaced before public distribution.
   The draft repository's blanket MIT file is deliberately not copied over them.
2. Confirm a dataset release decision for ViconStyle and its captions/processed files,
   including recording permissions and any relevant fitting-derived restrictions.
   Other datasets and scenes need their own acquisition/redistribution instructions.
3. Publish actual permitted data/weight archives and checksums, or provide real
   reconstruction instructions. No placeholder download URLs are invented here.
4. Validate the intended public checkpoints on representative scenes with the
   original success thresholds. The completed smoke checks do not certify reproduction
   of paper metrics or the remaining skills' training convergence.
5. Review code and documentation before publishing to the repository above.
   Keep motion data, scenes, weights, and local validation artifacts outside Git.
   No commits, pushes, remote visibility changes, or uploads were performed during validation.
