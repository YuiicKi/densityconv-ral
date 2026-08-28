#!/usr/bin/env python3
"""Matched-token diagnostic comparison using nuscenes-devkit metrics."""

import argparse
import json

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt_of_sample_tokens,
    load_prediction,
)
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricDataList, DetectionMetrics


def subset(boxes, tokens):
    out = EvalBoxes()
    for token in tokens:
        out.add_boxes(token, boxes[token])
    return out


def prepare(nusc, boxes, cfg):
    boxes = add_center_dist(nusc, boxes)
    return filter_eval_boxes(nusc, boxes, cfg.class_range, verbose=False)


def score(gt_boxes, pred_boxes, cfg):
    data = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            data.set(class_name, dist_th, accumulate(
                gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th
            ))

    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            metrics.add_label_ap(class_name, dist_th, calc_ap(
                data[(class_name, dist_th)], cfg.min_recall, cfg.min_precision
            ))
        for metric_name in TP_METRICS:
            md = data[(class_name, cfg.dist_th_tp)]
            if class_name == "traffic_cone" and metric_name in {"attr_err", "vel_err", "orient_err"}:
                value = np.nan
            elif class_name == "barrier" and metric_name in {"attr_err", "vel_err"}:
                value = np.nan
            else:
                value = calc_tp(md, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, value)
    result = metrics.serialize()
    result["prediction_count_filtered"] = sum(len(pred_boxes[t]) for t in pred_boxes.sample_tokens)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument("--dataroot", default="/mnt/ssd/nuscenes/")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = config_factory("detection_cvpr_2019")
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
    candidate, _ = load_prediction(args.candidate, cfg.max_boxes_per_sample, DetectionBox, verbose=False)
    reference_all, _ = load_prediction(args.reference, cfg.max_boxes_per_sample, DetectionBox, verbose=False)
    tokens = list(candidate.sample_tokens)
    reference = subset(reference_all, tokens)
    gt = load_gt_of_sample_tokens(nusc, tokens, DetectionBox, verbose=False)
    candidate = prepare(nusc, candidate, cfg)
    reference = prepare(nusc, reference, cfg)
    gt = prepare(nusc, gt, cfg)

    cand_score = score(gt, candidate, cfg)
    ref_score = score(gt, reference, cfg)
    result = {
        "diagnostic_subset": True,
        "sample_count": len(tokens),
        "ground_truth_count_filtered": sum(len(gt[t]) for t in tokens),
        "sample_tokens": tokens,
        "candidate": cand_score,
        "reference": ref_score,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    print(json.dumps({
        "samples": result["sample_count"],
        "gt": result["ground_truth_count_filtered"],
        "candidate": {
            "pred": cand_score["prediction_count_filtered"],
            "mAP": cand_score["mean_ap"],
            "NDS": cand_score["nd_score"],
        },
        "reference": {
            "pred": ref_score["prediction_count_filtered"],
            "mAP": ref_score["mean_ap"],
            "NDS": ref_score["nd_score"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
