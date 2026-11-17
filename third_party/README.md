# Third-party provenance

The retained physics/task/training/PoseLib code descends from the local ASE checkout
with origin `https://github.com/nv-tlabs/ASE`, commit
`87bbde5deb6a4f1cd8355b10d96550ca99ff8e6e`.
Its existing `LICENSE.txt` is preserved here as `ASE_LICENSE.txt`.
Per-file NVIDIA notices are preserved, including files with different notice text.
This record does not replace a per-file provenance and release-terms review.

Primary SIMS source snapshots consulted:

- `sims` experiment checkout: `f40ddd003900d026aba24dabe5422d43f3377caf` plus its working-tree assets.
- `sims-stylized_hsi` checkout: `3b50a83691a29161506db53a6a2839c9442f0c57`;
  its migrated code was still untracked, not part of that commit.
- `sims_iccv` supplies the older polygon helper behavior and original motion-data examples.

CLIP and rl-games remain external dependencies. Isaac Gym, pretrained CLIP weights,
SMPL/SMPL-X files, scene datasets, and motion datasets are not vendored in the code tree.
No blanket relicensing of these components is intended.
