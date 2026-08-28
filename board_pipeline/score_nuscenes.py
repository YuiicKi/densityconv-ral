#!/usr/bin/env python3
"""nuScenes official detection eval on RDK S100 (CPU, no CUDA needed).

Usage:
  MPLBACKEND=Agg /root/venv_eval/bin/python score_nuscenes.py <results.json> [--output_dir DIR]

Prints mAP / NDS / mATE / mASE / mAOE / mAVE / mAAE and writes metrics_summary.json
into <results>.metrics/ (or --output_dir).
"""
import argparse, json, os, time

parser = argparse.ArgumentParser()
parser.add_argument('results')
parser.add_argument('--output_dir', default=None)
parser.add_argument('--dataroot', default='/mnt/ssd/nuscenes/')
args = parser.parse_args()

out_dir = args.output_dir or (args.results + '.metrics')
os.makedirs(out_dir, exist_ok=True)

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory

t0 = time.time()
nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
cfg = config_factory('detection_cvpr_2019')
ev = NuScenesEval(nusc, config=cfg, result_path=args.results, eval_set='val',
                  output_dir=out_dir, verbose=False)
print(f'[setup {time.time()-t0:.0f}s] scoring {args.results} ...', flush=True)

t0 = time.time()
ev.main(render_curves=False)
print(f'[scored in {time.time()-t0:.0f}s]', flush=True)

s = json.load(open(os.path.join(out_dir, 'metrics_summary.json')))
print(f"\nmAP:  {s['mean_ap']:.4f}")
print(f"NDS:  {s['nd_score']:.4f}")
tp = s['tp_errors']
print(f"mATE: {tp['trans_err']:.4f}  mASE: {tp['scale_err']:.4f}  mAOE: {tp['orient_err']:.4f}"
      f"  mAVE: {tp['vel_err']:.4f}  mAAE: {tp['attr_err']:.4f}")
print(f"per-class AP (mean over dist thresholds):")
for cls in sorted(s['mean_dist_aps']):
    print(f"  {cls:20s} {s['mean_dist_aps'][cls]:.4f}")
print(f"summary: {os.path.join(out_dir, 'metrics_summary.json')}")
