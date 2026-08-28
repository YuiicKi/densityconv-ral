# Training commands (nuScenes multi-seed runs)

All nine models in Table I (3 architectures × seeds {7, 13, 59}) were launched
with the same command template on 2×A800:

```sh
python -m torch.distributed.launch --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT \
    train.py --launcher pytorch \
    --cfg_file cfgs/nuscenes_models/<CONFIG>.yaml \
    --extra_tag <ARCH>_rr<SEED> \
    --batch_size 8 --workers 12 \
    --fix_random_seed --seed <SEED>
```

with `<CONFIG>` one of:

| architecture | config |
|---|---|
| DensityConv | `cbgs_pillar0075_densepillar_centerpoint_densityconv.yaml` |
| Conv2d-only | `cbgs_pillar0075_densepillar_centerpoint.yaml` |
| SpConv (matched) | `cbgs_pillar0075_densepillar_centerpoint_spconv.yaml` |

and `<SEED>` ∈ {7, 13, 59}.

## Batch size

`--batch_size 8` is the **global** batch size: OpenPCDet's `train.py` divides
it by the number of GPUs, giving 4 per GPU on 2 GPUs (the training logs record
`total_batch_size: 8`, `batch_size 4` per process). The
`BATCH_SIZE_PER_GPU: 8` value in the YAMLs is the historical per-GPU default
and is overridden by the CLI flag; it is kept verbatim for byte-consistency
with the released training logs.

## Random seed

Stock OpenPCDet's `--fix_random_seed` seeds with a fixed internal constant.
Our `train.py` additionally accepts `--seed N` and forwards it to
`common_utils.set_random_seed(N)`, so `--fix_random_seed --seed 13` seeds all
of Python/NumPy/PyTorch with 13 (the training logs record both flags in the
argument dump). No other change to the stock training loop.

## Amplification baseline

The amplification variant (`*_sparseconv.yaml`, `SPARSE_CONV_TYPE:
sparse_aware`) is a single earlier run on 1 GPU with `--batch_size 4` and no
seed flag; it was not retrained for the revision. Its INT8 results reuse this
single checkpoint under three paired calibration seeds.

## Waymo

Waymo models (`configs/waymo/`) were trained for 30 epochs, single seed,
single-frame input, 0.1 m pillars; same launcher template.
