#!/usr/bin/env python3
"""
Export Conv2d-only model to ONNX. No mask input needed (NUM_SPARSE_STAGES=0).

Usage:
    cd ~/WORKDIR/tools && conda activate pcdet-v2-deploy
    python deploy/export_conv2d_only.py
"""
import os
import sys
import torch
import torch.nn as nn

ROOT = os.path.expanduser('~/WORKDIR')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools/deploy'))

from export_densepillar_onnx import build_model


class Conv2dOnlyWrapper(nn.Module):
    def __init__(self, backbone_3d, backbone_2d, dense_head):
        super().__init__()
        self.conv1 = backbone_3d.conv1
        self.conv2 = backbone_3d.conv2
        self.conv3 = backbone_3d.conv3
        self.conv4 = backbone_3d.conv4
        self.conv5 = backbone_3d.conv5
        self.bev_blocks = backbone_2d.blocks
        self.bev_deblocks = backbone_2d.deblocks
        self.shared_conv = dense_head.shared_conv
        self.heads_list = dense_head.heads_list

    def forward(self, bev_features):
        x1 = self.conv1(bev_features)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        ups = [self.bev_deblocks[0](x4)]
        x = self.bev_blocks[1](x5)
        ups.append(self.bev_deblocks[1](x))
        x = torch.cat(ups, dim=1)
        x = self.bev_blocks[0](x)
        x = self.shared_conv(x)
        outputs = []
        for head in self.heads_list:
            pred_dict = head(x)
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


def main():
    cfg_file = os.path.join(ROOT, 'tools/cfgs/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint.yaml')
    ckpt_file = os.path.join(ROOT, 'output/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint/conv2d_rr13/ckpt/latest_model.pth')
    output_path = os.path.join(ROOT, 'export/conv2d_only_rr13.onnx')

    model, _ = build_model(cfg_file, ckpt_file)
    wrapper = Conv2dOnlyWrapper(model.backbone_3d, model.backbone_2d, model.dense_head)
    wrapper.eval().cuda()

    dummy = torch.randn(1, 32, 1088, 1088).cuda()

    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=['bev_features'],
        output_names=[f'head_{i}' for i in range(6)],
        opset_version=11,
    )
    print(f'Exported: {output_path}')
    print(f'Size: {os.path.getsize(output_path) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
