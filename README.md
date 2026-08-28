# DensityConv — anonymous artifact repository (RA-L submission 26-2920)

Artifacts supporting *DensityConv: SpConv-Free 3D Detection for Robotic Edge
BPU*. Internal identifiers in historical logs differ from paper naming — see
`NAMING.md` for the mapping and the evidence-preservation policy.

## Layout

| directory | contents | paper reference |
|---|---|---|
| `pcdet/` | DensityConv implementation, backbone, training configs (nuScenes + Waymo), exact training commands (`pcdet/TRAINING.md`), ONNX export, calibration BEV rendering, full-val activation-statistics script (drop-in for OpenPCDet; see `pcdet/README.md`) | Sec. III |
| `calibration_ids/` | calibration frame indices, all 5 seeds × 3 budgets (`t2_s{seed}_{budget}_indices.txt`); identical IDs used for both models per seed/budget | Sec. IV-F, Table VI |
| `compiler_configs/` | per-run hb_compile `config.yaml`, `compile.log`, and per-node calibrated thresholds `quant_info.json` — fixed-percentile main matrix, calibration-method matrix (percentile/pure-max/KL), amplification variant, and FP32-exemption probe runs | Tables VI–VII, Sec. IV-G |
| `board_pipeline/` | RDK S100 C++ evaluation pipeline source (pillar VFE, BPU inference, rotated-IoU NMS), evaluation queue scripts, nuScenes scorer | Sec. IV-H, Table VIII |
| `profiling/` | runtime profiler outputs (`prof_dc/`, `prof_c2/`: single fused BPU segment, zero CPU inference cost), DDR hardware-counter logs (`ddr4k_*`, `hrut_ddr` over 4000-frame runs) with compiler schedule summaries, and per-frame serial-mode timing CSVs for Table VIII (`e2e_serial/`, 3 calibration seeds × 2 models × 6019 frames) | Sec. IV-H, Table VIII |

## Environment

- Training/evaluation: OpenPCDet, PyTorch, 2×A800 (global batch 8; the
  amplification baseline is a single-GPU batch-4 run, as stated in the paper)
- Compilation: Horizon OpenExplorer OE 3.2.0 (hb_compile 3.3.22, hbdk 4.2.11)
- Board: RDK S100 (80 TOPS BPU), RDK OS 4.0.5-Beta / HBRT 4.7.5
- nuScenes v1.0-trainval (10-sweep), Waymo Open Dataset v1.x (single-frame)

Training logs and model checkpoints are retained by the authors and available
upon request within the constraints of double-blind review.
