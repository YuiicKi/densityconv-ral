#!/usr/bin/env python
"""
Export DensePillar CenterPoint model to ONNX for RDK S100 deployment.

Boundary:
  - VFE + scatter run on host CPU (dynamic ops, not ONNX-exportable)
  - This script exports: Backbone stages (DensityConv + Dense) + FPN + CenterHead

Input:
  bev_features: (1, 32, 1088, 1088)  - pre-scattered BEV feature map
  bev_mask:     (1, 1, 1088, 1088)   - binary occupancy mask

Output:
  head_0 ~ head_5: per-head concatenated predictions
  (hm | center | center_z | dim | rot | vel)

Usage:
    cd ~/WORKDIR
    conda activate pcdet-v2-dist
    python tools/deploy/export_densepillar_onnx.py \
        --cfg_file tools/cfgs/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint_densityconv.yaml \
        --ckpt output/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint_densityconv/v1/ckpt/latest_model.pth \
        --output export/densepillar_centerpoint.onnx
"""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


class DensePillarONNXWrapper(nn.Module):
    """
    ONNX-exportable wrapper for DensePillar + FPN + CenterHead.

    Takes pre-scattered BEV features and binary mask as input,
    outputs raw detection head predictions.
    """

    def __init__(self, backbone_3d, backbone_2d, dense_head):
        super().__init__()

        # --- Backbone 3D: PillarDenseBackBone8x conv stages ---
        # Stages 1-2: SparseStage with DensityConv (DensityConv2d)
        self.conv1 = backbone_3d.conv1  # SparseStage(32->32, stride=1)
        self.conv2 = backbone_3d.conv2  # SparseStage(32->64, stride=2)
        # Stages 3-5: standard Conv2d
        self.conv3 = backbone_3d.conv3  # Sequential(64->128, stride=2)
        self.conv4 = backbone_3d.conv4  # Sequential(128->256, stride=2)
        self.conv5 = backbone_3d.conv5  # Sequential(256->256, stride=2)

        # --- Backbone 2D: BaseBEVBackboneV1 (FPN) ---
        self.bev_blocks = backbone_2d.blocks    # [block_0, block_1]
        self.bev_deblocks = backbone_2d.deblocks  # [deblock_0, deblock_1]

        # --- CenterHead ---
        self.shared_conv = dense_head.shared_conv
        self.heads_list = dense_head.heads_list

    def forward(self, bev_features, bev_mask):
        """
        Args:
            bev_features: (B, 32, H, W) scattered BEV tensor
            bev_mask: (B, 1, H, W) binary occupancy mask
        Returns:
            tuple of 6 tensors, one per detection head
        """
        mask = (bev_mask > 0.5).float()  # explicit binarization for INT8 deployment

        # === Stage 1-2: DensityConv stages ===
        x_conv1, mask = self.conv1(bev_features, mask)
        x_conv2, mask = self.conv2(x_conv1, mask)

        # === Stage 3-5: standard Conv2d ===
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        # === FPN ===
        ups = [self.bev_deblocks[0](x_conv4)]
        x = self.bev_blocks[1](x_conv5)
        ups.append(self.bev_deblocks[1](x))
        x = torch.cat(ups, dim=1)
        x = self.bev_blocks[0](x)

        # === CenterHead ===
        x = self.shared_conv(x)

        outputs = []
        for head in self.heads_list:
            pred_dict = head(x)
            # Concatenate: hm | center | center_z | dim | rot | vel
            head_out = torch.cat([
                pred_dict['hm'],
                pred_dict['center'],
                pred_dict['center_z'],
                pred_dict['dim'],
                pred_dict['rot'],
                pred_dict['vel'],
            ], dim=1)
            outputs.append(head_out)

        return tuple(outputs)


def fuse_conv_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d (for standard conv only)."""
    assert isinstance(conv, nn.Conv2d)
    assert isinstance(bn, nn.BatchNorm2d)

    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        conv.kernel_size, stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=conv.groups, bias=True,
    ).to(conv.weight.device)

    # BN params
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)

    fused.weight.data = conv.weight * scale.reshape(-1, 1, 1, 1)
    if conv.bias is not None:
        fused.bias.data = (conv.bias - mean) * scale + beta
    else:
        fused.bias.data = -mean * scale + beta

    return fused


def fuse_bn_in_sequential(seq):
    """Fuse Conv2d+BN pairs in a nn.Sequential."""
    modules = list(seq.children())
    fused_modules = []
    i = 0
    while i < len(modules):
        if isinstance(modules[i], nn.Conv2d) and i + 1 < len(modules) and isinstance(modules[i + 1], nn.BatchNorm2d):
            fused_modules.append(fuse_conv_bn(modules[i], modules[i + 1]))
            i += 2
        elif isinstance(modules[i], nn.Sequential):
            fused_modules.append(fuse_bn_in_sequential(modules[i]))
            i += 1
        else:
            fused_modules.append(modules[i])
            i += 1
    return nn.Sequential(*fused_modules)


def fold_bn_recursive(module):
    """Recursively fold BatchNorm into Conv2d where possible."""
    # Handle nn.Sequential
    if isinstance(module, nn.Sequential):
        return fuse_bn_in_sequential(module)

    # Recurse into children
    for name, child in module.named_children():
        folded = fold_bn_recursive(child)
        if folded is not child:
            setattr(module, name, folded)

    return module


def build_model(cfg_file, ckpt_path):
    """Build and load the full OpenPCDet model."""
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()

    # We need the dataset to build the model (for grid_size, etc.)
    # Use a minimal dataset
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set
    )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    model.cuda()

    print(f"[INFO] Model loaded from {ckpt_path}")
    print(f"[INFO] Grid size: {test_set.grid_size}")
    print(f"[INFO] Voxel size: {test_set.voxel_size}")

    return model, test_set


def export_onnx(model, test_set, output_path, opset=11):
    """Export the 2D pipeline to ONNX."""

    # Extract sub-modules
    backbone_3d = model.backbone_3d   # PillarDenseBackBone8x
    backbone_2d = model.backbone_2d   # BaseBEVBackboneV1
    dense_head = model.dense_head     # CenterHead

    # Build ONNX wrapper
    wrapper = DensePillarONNXWrapper(backbone_3d, backbone_2d, dense_head)
    wrapper.eval()
    wrapper.cuda()

    # BN folding for standard Conv2d stages (conv3-5, FPN, CenterHead)
    # NOTE: DensityConv stages keep BN separate (output scaled by density)
    print("[INFO] Folding BatchNorm into Conv2d for dense stages...")
    wrapper.conv3 = fold_bn_recursive(wrapper.conv3)
    wrapper.conv4 = fold_bn_recursive(wrapper.conv4)
    wrapper.conv5 = fold_bn_recursive(wrapper.conv5)
    wrapper.bev_blocks = nn.ModuleList([fold_bn_recursive(b) for b in wrapper.bev_blocks])
    wrapper.bev_deblocks = nn.ModuleList([fold_bn_recursive(b) for b in wrapper.bev_deblocks])
    wrapper.shared_conv = fold_bn_recursive(wrapper.shared_conv)
    for i, head in enumerate(wrapper.heads_list):
        for attr_name in head.sep_head_dict:
            fc = getattr(head, attr_name)
            setattr(head, attr_name, fold_bn_recursive(fc))

    # Prepare dummy inputs
    nx = int(test_set.grid_size[0])
    ny = int(test_set.grid_size[1])
    in_channels = backbone_3d.in_channels  # 32

    print(f"[INFO] BEV grid: ({in_channels}, {ny}, {nx})")

    bev_features = torch.randn(1, in_channels, ny, nx, device='cuda')
    bev_mask = (torch.rand(1, 1, ny, nx, device='cuda') > 0.9).float()

    # Verify forward pass
    print("[INFO] Running forward pass to verify...")
    with torch.no_grad():
        outputs = wrapper(bev_features, bev_mask)
        for i, out in enumerate(outputs):
            print(f"  head_{i}: shape={list(out.shape)}")

    # Export to ONNX
    print(f"[INFO] Exporting to ONNX (opset={opset})...")

    output_names = [f'head_{i}' for i in range(len(outputs))]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    torch.onnx.export(
        wrapper,
        (bev_features, bev_mask),
        output_path,
        input_names=['bev_features', 'bev_mask'],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # Static shapes for BPU
    )

    # Verify ONNX
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] ONNX exported: {output_path} ({file_size_mb:.1f} MB)")
    print(f"[INFO] Inputs:  bev_features=(1,{in_channels},{ny},{nx}), bev_mask=(1,1,{ny},{nx})")
    print(f"[INFO] Outputs: {output_names}")

    return output_path


def generate_calibration_data(model, test_set, output_dir, num_samples=50):
    """
    Generate calibration data for INT8 quantization.

    Runs VFE + scatter on real data, saves BEV features + mask as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)

    backbone_3d = model.backbone_3d
    model.eval()
    model.cuda()

    print(f"[INFO] Generating {num_samples} calibration samples...")

    bev_list = []
    mask_list = []

    for i in range(min(num_samples, len(test_set))):
        data = test_set[i]
        # Collate single sample
        from pcdet.datasets import DatasetTemplate
        batch = test_set.collate_batch([data])
        for k, v in batch.items():
            if isinstance(v, np.ndarray) and v.dtype.kind in ('f', 'i', 'u'):
                batch[k] = torch.from_numpy(v).cuda()
            elif isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        with torch.no_grad():
            # Run VFE
            batch = model.vfe(batch)
            # Run scatter + get BEV features
            batch['batch_size'] = 1
            spatial_features = backbone_3d.scatter_pillars_to_bev(
                1, batch['pillar_features'], batch['pillar_coords']
            )
            mask = (spatial_features.abs().sum(dim=1, keepdim=True) > 0).float()

        bev_np = spatial_features.cpu().numpy()
        mask_np = mask.cpu().numpy()

        np.save(os.path.join(output_dir, f'bev_{i:04d}.npy'), bev_np)
        np.save(os.path.join(output_dir, f'mask_{i:04d}.npy'), mask_np)

        if i % 10 == 0:
            print(f"  [{i}/{num_samples}] BEV occupancy: {mask_np.mean():.4f}")

    print(f"[SUCCESS] Calibration data saved to {output_dir}/")


def benchmark_pytorch(model, test_set, num_runs=100):
    """Benchmark PyTorch inference latency on GPU."""
    backbone_3d = model.backbone_3d
    nx = int(test_set.grid_size[0])
    ny = int(test_set.grid_size[1])
    in_ch = backbone_3d.in_channels

    wrapper = DensePillarONNXWrapper(backbone_3d, model.backbone_2d, model.dense_head)
    wrapper.eval().cuda()

    bev = torch.randn(1, in_ch, ny, nx, device='cuda')
    mask = (torch.rand(1, 1, ny, nx, device='cuda') > 0.9).float()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = wrapper(bev, mask)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = wrapper(bev, mask)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_runs * 1000

    print(f"[BENCHMARK] PyTorch GPU latency: {elapsed:.2f} ms (avg over {num_runs} runs)")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', required=True, help='Model config YAML')
    parser.add_argument('--ckpt', required=True, help='Checkpoint .pth path')
    parser.add_argument('--output', default='export/densepillar_centerpoint.onnx')
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--gen_calib', action='store_true',
                        help='Generate INT8 calibration data')
    parser.add_argument('--calib_dir', default='export/calibration_data')
    parser.add_argument('--calib_samples', type=int, default=50)
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark PyTorch latency')
    args = parser.parse_args()

    model, test_set = build_model(args.cfg_file, args.ckpt)

    # Export ONNX
    export_onnx(model, test_set, args.output, args.opset)

    # Optionally generate calibration data
    if args.gen_calib:
        generate_calibration_data(model, test_set, args.calib_dir, args.calib_samples)

    # Optionally benchmark
    if args.benchmark:
        benchmark_pytorch(model, test_set)


if __name__ == '__main__':
    main()
