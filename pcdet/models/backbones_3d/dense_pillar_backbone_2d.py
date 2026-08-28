from functools import partial

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super().__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class PillarDenseBackBone8x(nn.Module):
    """
    Dense 2D BEV backbone for PillarNet without spconv.

    - Takes pillar_features + pillar_coords and scatters them to a dense BEV canvas.
    - Then applies a small 2D ResNet-style backbone to produce multi-scale 2D features
      compatible with BaseBEVBackboneV1.
    - This is intended for platforms where spconv is not available, at the cost of
      higher dense computation.

    Supports optional sparse convolution for the first N stages via config:
      SPARSE_CONV_TYPE: 'none' | 'partial' (DensityConv) | 'sparse_aware' (amplification) | 'spconv'
      NUM_SPARSE_STAGES: 2  (default, stages 1-2 use sparse conv)
    """

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # grid_size: [Nx, Ny, Nz] or similar, we only need X/Y for BEV.
        self.nx = int(grid_size[0])
        self.ny = int(grid_size[1])
        self.in_channels = input_channels  # e.g., 32 from DynamicPillarVFESimple2D

        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        # Sparse convolution config
        sparse_type = model_cfg.get('SPARSE_CONV_TYPE', 'none')
        self.sparse_type = sparse_type
        num_sparse = model_cfg.get('NUM_SPARSE_STAGES', 2) if sparse_type != 'none' else 0
        self.num_sparse_stages = num_sparse

        sparse_cls = None
        if sparse_type == 'spconv':
            # Native spconv path: use SpconvStage with BatchNorm1d
            from pcdet.models.backbones_2d.sparse_conv_modules import SpconvStage
            spconv_norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            self.sparse_shape = [self.ny, self.nx]

            if num_sparse >= 1:
                self.conv1 = SpconvStage(self.in_channels, 32, kernel_size=3,
                                         stride=1, num_blocks=1,
                                         norm_fn=spconv_norm_fn, stage_id=1)
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1,
                              padding=1, bias=False),
                    norm_fn(32), nn.ReLU(),
                    BasicBlock(32, 32, norm_fn=norm_fn),
                )

            if num_sparse >= 2:
                self.conv2 = SpconvStage(32, 64, kernel_size=3, stride=2,
                                         num_blocks=2,
                                         norm_fn=spconv_norm_fn, stage_id=2)
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_fn(64), nn.ReLU(),
                    BasicBlock(64, 64, norm_fn=norm_fn),
                    BasicBlock(64, 64, norm_fn=norm_fn),
                )

            if num_sparse >= 3:
                self.conv3 = SpconvStage(64, 128, kernel_size=3, stride=2,
                                         num_blocks=2,
                                         norm_fn=spconv_norm_fn, stage_id=3)
            else:
                self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_fn(128), nn.ReLU(),
                    BasicBlock(128, 128, norm_fn=norm_fn),
                    BasicBlock(128, 128, norm_fn=norm_fn),
                )

        elif sparse_type != 'none':
            # DensityConv / SparseAwareConv path (mask-based)
            from pcdet.models.backbones_2d.sparse_conv_modules import (
                DensityConv2d, SparseAwareConv2d, DACASparseConv2d, SparseStage
            )
            from functools import partial as _partial
            if sparse_type == 'partial':
                sparse_cls = DensityConv2d
            elif sparse_type == 'sparse_aware':
                sparse_cls = SparseAwareConv2d
            elif sparse_type == 'daca':
                _daca_thresh = float(model_cfg.get('DACA_THRESHOLD', 0.0))
                _daca_src = str(model_cfg.get('DACA_GATE_SOURCE', 'mask_sum'))
                sparse_cls = _partial(DACASparseConv2d,
                                      threshold=_daca_thresh,
                                      gate_source=_daca_src)
            else:
                raise ValueError(f"Unknown SPARSE_CONV_TYPE: {sparse_type}")

            if num_sparse >= 1:
                self.conv1 = SparseStage(sparse_cls, self.in_channels, 32,
                                         kernel_size=3, stride=1, num_blocks=1,
                                         norm_fn=norm_fn)
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1,
                              padding=1, bias=False),
                    norm_fn(32), nn.ReLU(),
                    BasicBlock(32, 32, norm_fn=norm_fn),
                )

            if num_sparse >= 2:
                self.conv2 = SparseStage(sparse_cls, 32, 64,
                                         kernel_size=3, stride=2, num_blocks=2,
                                         norm_fn=norm_fn)
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_fn(64), nn.ReLU(),
                    BasicBlock(64, 64, norm_fn=norm_fn),
                    BasicBlock(64, 64, norm_fn=norm_fn),
                )

            if num_sparse >= 3:
                self.conv3 = SparseStage(sparse_cls, 64, 128,
                                         kernel_size=3, stride=2, num_blocks=2,
                                         norm_fn=norm_fn)
            else:
                self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                    norm_fn(128), nn.ReLU(),
                    BasicBlock(128, 128, norm_fn=norm_fn),
                    BasicBlock(128, 128, norm_fn=norm_fn),
                )

        else:
            # Pure Conv2d baseline (no sparse conv)
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1,
                          padding=1, bias=False),
                norm_fn(32), nn.ReLU(),
                BasicBlock(32, 32, norm_fn=norm_fn),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_fn(64), nn.ReLU(),
                BasicBlock(64, 64, norm_fn=norm_fn),
                BasicBlock(64, 64, norm_fn=norm_fn),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                norm_fn(128), nn.ReLU(),
                BasicBlock(128, 128, norm_fn=norm_fn),
                BasicBlock(128, 128, norm_fn=norm_fn),
            )

        # Stage 4: stride-2 downsample, 256 channels (always standard)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1,
                      bias=False),
            norm_fn(256),
            nn.ReLU(),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        # Stage 5: extra stride-2 downsample (always standard)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1,
                      bias=False),
            norm_fn(256),
            nn.ReLU(),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256

        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256,
        }

    def scatter_pillars_to_bev(self, batch_size, pillar_features, pillar_coords):
        """
        ONNX-compatible scatter using torch.scatter.

        pillar_features: (N, C_in)
        pillar_coords: (N, 3) [batch_idx, y, x]
        Returns:
            spatial_features: (B, C_in, H, W)
        """
        B, C, H, W = batch_size, self.in_channels, self.ny, self.nx
        device = pillar_features.device

        # Flatten spatial dimensions for scatter operation
        spatial_features_flat = pillar_features.new_zeros(B * H * W, C, device=device)

        # Compute 1D indices from [batch, y, x] coordinates
        coords = pillar_coords.long()
        batch_idx = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        # Convert 3D indices to 1D: flat_idx = batch * (H*W) + y * W + x
        flat_idx = batch_idx * (H * W) + y * W + x  # (N,)
        flat_idx = flat_idx.unsqueeze(1).expand(-1, C)  # (N, C)

        # Scatter operation (ONNX supported)
        spatial_features_flat.scatter_(0, flat_idx, pillar_features)

        # Reshape back to BEV grid.
        # spatial_features_flat is laid out as (B*H*W, C) with flat index = b*(H*W) + y*W + x.
        # The correct contiguous view is (B, H, W, C), then permute to (B, C, H, W).
        spatial_features = spatial_features_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return spatial_features

    def forward(self, batch_dict):
        pillar_features = batch_dict['pillar_features']
        pillar_coords = batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']

        if self.sparse_type == 'spconv':
            # Native spconv path: build SparseConvTensor directly, no scatter
            import spconv.pytorch as spconv_lib
            sp_tensor = spconv_lib.SparseConvTensor(
                features=pillar_features,
                indices=pillar_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )

            # Only stages 1-3 can be spconv; stages 4-5 are always dense
            n_sp = min(self.num_sparse_stages, 3)
            stages = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
            results = []
            x = sp_tensor

            for i, stage in enumerate(stages):
                if i < n_sp:
                    x = stage(x)
                else:
                    if i == n_sp:
                        x = x.dense()
                    x = stage(x)
                results.append(x)

            x_conv1, x_conv2, x_conv3, x_conv4, x_conv5 = results

        elif self.num_sparse_stages > 0:
            # Mask-based sparse conv path (DensityConv / SparseAwareConv)
            spatial_features = self.scatter_pillars_to_bev(
                batch_size, pillar_features, pillar_coords
            )
            mask = (spatial_features.abs().sum(dim=1, keepdim=True) > 0).float()

            # Stage 1
            if self.num_sparse_stages >= 1:
                x_conv1, mask = self.conv1(spatial_features, mask)
            else:
                x_conv1 = self.conv1(spatial_features)

            # Stage 2
            if self.num_sparse_stages >= 2:
                x_conv2, mask = self.conv2(x_conv1, mask)
            else:
                x_conv2 = self.conv2(x_conv1)

            # Stage 3
            if self.num_sparse_stages >= 3:
                x_conv3, mask = self.conv3(x_conv2, mask)
            else:
                x_conv3 = self.conv3(x_conv2)

            # Stages 4-5 (always standard, features are sufficiently dense)
            x_conv4 = self.conv4(x_conv3)
            x_conv5 = self.conv5(x_conv4)

        else:
            # Baseline: all standard Conv2d
            spatial_features = self.scatter_pillars_to_bev(
                batch_size, pillar_features, pillar_coords
            )
            x_conv1 = self.conv1(spatial_features)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)
            x_conv5 = self.conv5(x_conv4)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict
