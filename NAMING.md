# Naming convention

Historical artifacts in this repository (compile logs, per-node threshold
dumps `quant_info.json`, run directory names, profiler outputs) predate the
paper's final naming and use internal identifiers. They are preserved
**verbatim as evidence** — only absolute user paths were anonymized in
shell/python scripts; no log or JSON content was edited.

| internal identifier | paper name |
|---|---|
| `partialconv`, `PartialConv2d`, `dc`, config keyword `partial` | **DensityConv** (attenuation operator, Eq. (2)) |
| `sparseaware`, `SparseAwareConv2d`, `amp`, `ampv1`, config keyword `sparse_aware` | amplification baseline (inverse-occupancy scaling) |
| `conv2d`, `c2`, config keyword `none` | Conv2d-only baseline |
| `spconv` | matched SpConv baseline |
| `rr7 / rr13 / rr59` | training seeds 7 / 13 / 59 |
| `s42 / s1 / s2 / s7 / s59` | calibration seeds 42 / 1 / 2 / 7 / 59 |
| `maxpct_pc_asym` | max-percentile (p=0.99995), per-channel, asymmetric |
| `maxpct_pt_asym` | max-percentile, per-tensor, asymmetric |
| `max_pc_asym` | pure-max, per-channel, asymmetric |
| `kl` | KL calibration |

In the code shipped here, the operator class has been renamed to
`DensityConv2d` to match the paper; the config keyword `partial` is kept so
that released configs remain byte-compatible with the training logs and
checkpoints that reference them.

A note on the compile-stage node tables: the calibration-stage `compile.log`
node table conservatively marks a few deep residual-block nodes as `CPU` (in
both models, none on the mask path). That table describes the intermediate
PTQ model; backend compilation folds these nodes back into the BPU graph.
The runtime profiler logs in `profiling/` confirm each compiled model
executes as a single fused BPU segment with zero CPU inference cost.
