# Distance-binned nuScenes detection eval, reusing the official nuscenes-devkit
# DetectionEval end-to-end. No metric logic is reimplemented: boxes are filtered
# by ego distance into bins, then the official evaluate() runs unchanged.
# Validation: the [0,max) bin must reproduce metrics_summary.json exactly.
import copy
import json
import os
import statistics as st

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

DATAROOT = os.path.expanduser('~/WORKDIR/data/nuscenes')
BASE = os.path.expanduser('~/WORKDIR/output/nuscenes_models')
OUT_DIR = os.path.expanduser('~/WORKDIR/tmp_nusc_range_eval_out')
SUMMARY_PATH = os.path.expanduser('~/WORKDIR/nusc_range_eval_summary.json')

SEEDS = ['rr7', 'rr13', 'rr59']
METHOD_DIRS = [
    ('conv2d', 'cbgs_pillar0075_densepillar_centerpoint'),
    ('dc', 'cbgs_pillar0075_densepillar_centerpoint_partialconv'),
    ('spconv', 'cbgs_pillar0075_densepillar_centerpoint_spconv'),
]
MODELS = {}
for meth, d in METHOD_DIRS:
    for s in SEEDS:
        MODELS[f'{meth}_{s}'] = (
            f'{BASE}/{d}/{meth}_{s}/eval/eval_with_train/epoch_20/val/final_result/data/results_nusc.json'
        )

# (lo, hi); hi=None means the per-class default range from detection_cvpr_2019
# (car/truck/bus/trailer/CV 50 m, ped/moto/bicycle 40 m, cone/barrier 30 m).
# [0,max) is a pure validation bin: must match the official metrics_summary.json.
BINS = [(0.0, None), (0.0, 20.0), (20.0, 30.0), (30.0, None)]

cfg = config_factory('detection_cvpr_2019')
DEFAULT_RANGE = cfg.class_range


def bin_tag(lo, hi):
    return '[%d,%s)' % (lo, 'max' if hi is None else '%d' % hi)


def included_classes(lo, hi):
    out = []
    for c, dflt in DEFAULT_RANGE.items():
        upper = dflt if hi is None else min(hi, dflt)
        if upper > lo:
            out.append(c)
    return out


nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)

summary = {}
for name, path in MODELS.items():
    if not os.path.exists(path):
        print('!! missing %s -- skipped' % path, flush=True)
        continue
    ev = DetectionEval(nusc, config=cfg, result_path=path, eval_set='val',
                       output_dir=OUT_DIR, verbose=False)
    pred0, gt0 = ev.pred_boxes, ev.gt_boxes
    for lo, hi in BINS:
        pred, gt = copy.deepcopy(pred0), copy.deepcopy(gt0)
        if not (lo == 0.0 and hi is None):  # validation bin keeps boxes untouched
            for eb in (pred, gt):
                for tok in eb.boxes:
                    eb.boxes[tok] = [
                        b for b in eb.boxes[tok]
                        if b.ego_dist >= lo and (
                            hi is None or b.ego_dist < min(hi, DEFAULT_RANGE[b.detection_name])
                        )
                    ]
        ev.pred_boxes, ev.gt_boxes = pred, gt
        metrics, _ = ev.evaluate()
        aps = metrics.serialize()['mean_dist_aps']
        incl = included_classes(lo, hi)
        m = float(np.mean([aps[c] for c in incl]))
        tag = bin_tag(lo, hi)
        print('%s %s mAP(%dcls)=%.4f  %s' % (
            name, tag, len(incl), m,
            ' '.join('%s:%.4f' % (c, aps[c]) for c in incl)), flush=True)
        summary.setdefault(name, {})[tag] = {'mAP': m, 'aps': {c: aps[c] for c in incl}}
    ev.pred_boxes, ev.gt_boxes = pred0, gt0

with open(SUMMARY_PATH, 'w') as f:
    json.dump(summary, f, indent=1)

print('\n=== mean±std over seeds (mAP x100) ===', flush=True)
for meth, _ in METHOD_DIRS:
    for lo, hi in BINS:
        tag = bin_tag(lo, hi)
        vals = [summary[f'{meth}_{s}'][tag]['mAP'] for s in SEEDS if f'{meth}_{s}' in summary]
        if len(vals) >= 2:
            print('%-8s %-9s %.2f±%.2f' % (meth, tag, st.mean(vals) * 100, st.stdev(vals) * 100), flush=True)
print('saved: %s' % SUMMARY_PATH)
