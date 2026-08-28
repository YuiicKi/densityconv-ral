#!/usr/bin/env python3
"""Diagnostic nuScenes detection score for the exact sample tokens in a result JSON.

This uses nuscenes-devkit's official loading, filtering, matching, AP and TP
implementations, but evaluates only the sample tokens present in the JSON.
It is a diagnostic subset score, not a replacement for the official val score.
"""

import argparse
import json
import math
import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionMetricDataList,
    DetectionMetrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--dataroot", default="/mnt/ssd/nuscenes/")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = config_factory("detection_cvpr_2019")
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)

    pred_boxes, meta = load_prediction(
        args.results, cfg.max_boxes_per_sample, DetectionBox, verbose=False
    )
    tokens = list(pred_boxes.sample_tokens)
    if not tokens:
        raise SystemExit("No sample tokens in prediction JSON")

    gt_all = load_gt(nusc, "val", DetectionBox, verbose=False)
    gt_boxes = EvalBoxes()
    for token in tokens:
        if token not in gt_all.sample_tokens:
            raise SystemExit(f"Prediction token is not in nuScenes val: {token}")
        gt_boxes.add_boxes(token, gt_all[token])

    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=False)

    metric_data = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data.set(class_name, dist_th, md)

    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = metric_data[(class_name, dist_th)]
            metrics.add_label_ap(
                class_name, dist_th, calc_ap(md, cfg.min_recall, cfg.min_precision)
            )
        for metric_name in TP_METRICS:
            md = metric_data[(class_name, cfg.dist_th_tp)]
            if class_name == "traffic_cone" and metric_name in {
                "attr_err",
                "vel_err",
                "orient_err",
            }:
                tp = np.nan
            elif class_name == "barrier" and metric_name in {"attr_err", "vel_err"}:
                tp = np.nan
            else:
                tp = calc_tp(md, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    summary = metrics.serialize()
    summary["diagnostic_subset"] = True
    summary["sample_count"] = len(tokens)
    summary["prediction_count"] = sum(len(pred_boxes[token]) for token in tokens)
    summary["ground_truth_count"] = sum(len(gt_boxes[token]) for token in tokens)
    summary["meta"] = meta
    summary["sample_tokens"] = tokens

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print(
        f"samples={len(tokens)} predictions={summary['prediction_count']} "
        f"gt={summary['ground_truth_count']}"
    )
    print(f"mAP={summary['mean_ap']:.10f} NDS={summary['nd_score']:.10f}")
    for name in sorted(summary["mean_dist_aps"]):
        value = summary["mean_dist_aps"][name]
        print(f"{name:20s} {value:.10f}" if math.isfinite(value) else f"{name:20s} nan")


if __name__ == "__main__":
    main()
