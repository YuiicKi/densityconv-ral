# DensityConv — minimal code drop-in for OpenPCDet

This directory contains the **incremental code** that adds DensityConv to a
compatible [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) installation.
It is not a standalone training framework; the host framework provides
`PillarNet`, `DynamicPillarVFESimple2D`, `BaseBEVBackboneV1`, `CenterHead`,
data loading, and training entry points (`tools/train.py`, `tools/test.py`).

## Contents

| file | purpose |
|---|---|
| `models/backbones_2d/sparse_conv_modules.py` | DensityConv2d (paper operator), SparseAwareConv2d (amplification baseline), SpConv stage wrappers |
| `models/backbones_3d/dense_pillar_backbone_2d.py` | PillarDenseBackBone8x (5-level backbone; Levels 1–2 selectable operator) |
| `configs/nuscenes/*.yaml` | training configs: DensityConv / Conv2d-only / matched SpConv / amplification |
| `configs/waymo/*.yaml` | Waymo configs (DensityConv, Conv2d-only) |
| `configs/dataset_configs/*.yaml` | base dataset configs referenced by the model configs |
| `deploy/export_densepillar_onnx.py` | ONNX export (Greater+Cast mask binarization; DensityConv path) |
| `deploy/export_conv2d_rr13.py` | ONNX export for the Conv2d-only model |
| `deploy/gen_dc_bev_for_step7.py`, `deploy/gen_conv2d_bev_for_step7.py` | calibration BEV/mask rendering from training-sample IDs (paired calibration sets) |
| `deploy/compile_*.yaml` | example hb_compile configs (the fixed-percentile configs used for the paper's main results are in `../compiler_configs/`) |
| `patches/register_densepillar.patch` | adds the backbone import/registration to the host framework |

## Placement

| this directory | OpenPCDet location |
|---|---|
| `models/backbones_2d/sparse_conv_modules.py` | `pcdet/models/backbones_2d/` |
| `models/backbones_3d/dense_pillar_backbone_2d.py` | `pcdet/models/backbones_3d/` |
| `configs/nuscenes/*.yaml` | `tools/cfgs/nuscenes_models/` |
| `configs/waymo/*.yaml` | `tools/cfgs/waymo_models/` |
| `configs/dataset_configs/*.yaml` | `tools/cfgs/dataset_configs/` |
| `deploy/*.py` | `tools/deploy/` |

Register the backbone by applying `patches/register_densepillar.patch`
(`git apply --check` first), or add the import and `__all__` entry to
`pcdet/models/backbones_3d/__init__.py` manually:

```python
from .dense_pillar_backbone_2d import PillarDenseBackBone8x
# add to the existing __all__ dict:
'PillarDenseBackBone8x': PillarDenseBackBone8x,
```

## Notes

- The operator is selected per config via `SPARSE_CONV_TYPE`:
  `partial` = **DensityConv** (paper operator), `sparse_aware` = amplification
  baseline, `spconv` = matched SpConv baseline, `none` = Conv2d-only.
  The config keyword `partial` is kept for consistency with the released
  training logs and checkpoints; see `../NAMING.md`.
- Calibration BEV generation takes explicit `--pcdet_root`, `--cfg_file`,
  `--ckpt`, `--indices_file`, `--output_dir`. The paired DC/Conv2d calibration
  sets use the same sample-ID files (`../calibration_ids/`) but each model's
  own config and checkpoint.
- `gen_dc_bev_for_step7.py` points test infos at
  `nuscenes_infos_10sweeps_train.pkl`; `training=False` disables training-mode
  augmentation, it does not select the validation split. Output: `bev/` and
  `mask/` subdirectories.
- Path defaults inside the deploy scripts were anonymized to `~/WORKDIR`;
  pass explicit arguments when running.
- The amplification config (`*_sparseconv.yaml`) reproduces the single-run
  baseline; its actual launch parameters (single-GPU, batch 4) override the
  YAML defaults, as recorded in the paper.
