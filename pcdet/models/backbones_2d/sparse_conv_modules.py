import torch
import torch.nn as nn
from functools import partial


class DensityConv2d(nn.Module):
    """
    DensityConv (paper operator): density-scaled convolution for sparse BEV.
    Extends the masked-propagation mechanism of Partial Convolution
    (Liu et al., ECCV 2018) with the opposite, bounded scaling direction.

    Normalization: out = Conv(x * mask) * (mask_sum / K^2)
    - mask_sum: number of valid pixels in receptive field
    - K^2: total pixels in kernel window
    - Scale factor = mask_sum / K^2 (occupancy ratio, always <= 1)
    - In sparse regions (low mask_sum), output is suppressed
    - Mask update: any valid pixel in receptive field -> output valid
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.register_buffer('k_sq', torch.tensor(float(k * k)))

        # Fixed all-1 kernel for counting valid pixels in receptive field
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride=stride,
                                   padding=padding, bias=False)
        nn.init.ones_(self.mask_conv.weight)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

    def forward(self, x, mask):
        """
        Args:
            x: (B, C, H, W) input features
            mask: (B, 1, H, W) binary occupancy mask
        Returns:
            out: (B, C_out, H', W') normalized output
            new_mask: (B, 1, H', W') updated mask
        """
        out = self.conv(x * mask)

        with torch.no_grad():
            mask_sum = self.mask_conv(mask)

        # Occupancy ratio scaling: mask_sum / K^2
        # Dense area (mask_sum ~ K^2): scale ~ 1, no change
        # Sparse area (mask_sum ~ 1): scale ~ 1/K^2, output suppressed
        scale = mask_sum / self.k_sq
        out = out * scale

        new_mask = (mask_sum > 0).float()
        return out, new_mask


class SparseAwareConv2d(nn.Module):
    """
    Sparse-Aware Convolution (ours).

    Normalization: out = Conv(x * M) * K^2 / clamp(valid_count, min=1)
    - valid_count: number of valid pixels in receptive field (fixed all-1 kernel)
    - K^2: total pixels in kernel window (fixed constant)
    - Scale factor = K^2 / valid_count (inverse occupancy, always >= 1)
    - In sparse regions (low valid_count), output is amplified to compensate
    - MaskConv weights are fixed and do NOT participate in training

    Amplification baseline compared in the paper (inverse-occupancy scaling,
    as in Partial Convolution, Liu et al., ECCV 2018):
    - DensityConv: scale = mask_sum / K^2 <= 1, bounded attenuation
    - This variant: scale = K^2 / valid_count >= 1, amplifies sparse regions
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.register_buffer('k_sq', torch.tensor(float(k * k)))

        # Fixed mask convolution (non-trainable)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride=stride,
                                   padding=padding, bias=False)
        nn.init.ones_(self.mask_conv.weight)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

    def forward(self, x, mask):
        """
        Args:
            x: (B, C, H, W) input features
            mask: (B, 1, H, W) binary occupancy mask
        Returns:
            out: (B, C_out, H', W') normalized output
            new_mask: (B, 1, H', W') updated mask
        """
        out_raw = self.conv(x * mask)

        with torch.no_grad():
            valid_count = self.mask_conv(mask)

        # Inverse occupancy normalization: K^2 / valid_count
        # Dense area (valid_count ~ K^2): scale ~ 1, no change
        # Sparse area (valid_count ~ 1): scale ~ K^2, amplifies output
        out = out_raw * self.k_sq / valid_count.clamp(min=1)

        new_mask = (valid_count > 0).float()
        return out, new_mask





class DACASparseConv2d(nn.Module):
    """
    DACA with configurable gate source for paper #2 attractor experiments.
    Output: out = Conv(x * mask) * gate, where gate is binary.

    Gate sources (yaml DACA_GATE_SOURCE):
      - "mask_sum" (default): gate = (mask_sum > threshold * k^2)
      - "sobel": gate = top-K of |Sobel(x*mask)|, K matched to mask_sum>0 coverage
      - "energy": gate = top-K of avg_pool((x*mask)^2), coverage matched
      - "random": gate = random binary at coverage matching mask_sum>0
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False,
                 threshold=0.0, gate_source="mask_sum"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.register_buffer("k_sq", torch.tensor(float(k * k)))
        self.register_buffer("thresh_count", torch.tensor(float(threshold) * float(k * k)))
        self.threshold = float(threshold)
        self.gate_source = gate_source
        self.stride_val = stride if isinstance(stride, int) else stride[0]

        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride=stride,
                                   padding=padding, bias=False)
        nn.init.ones_(self.mask_conv.weight)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

        if gate_source == "sobel":
            sx = torch.tensor([[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0],
                               [-1.0, 0.0, 1.0]]) / 4.0
            sy = sx.t().contiguous()
            self.register_buffer("sobel_x", sx.view(1, 1, 3, 3))
            self.register_buffer("sobel_y", sy.view(1, 1, 3, 3))

    @torch.no_grad()
    def _coverage_matched_gate(self, score, ref_active):
        B = score.shape[0]
        flat_score = score.view(B, -1)
        n_active = ref_active.view(B, -1).sum(dim=1).long()
        gate_flat = torch.zeros_like(flat_score)
        for b in range(B):
            k = max(1, int(n_active[b].item()))
            if k >= flat_score.shape[1]:
                gate_flat[b] = 1.0
            else:
                kth = flat_score[b].topk(k, sorted=False).values.min()
                gate_flat[b] = (flat_score[b] >= kth).float()
        return gate_flat.view_as(score)

    def forward(self, x, mask):
        out = self.conv(x * mask)

        with torch.no_grad():
            mask_sum = self.mask_conv(mask)
            ref_active = (mask_sum > 0).float()

        if self.gate_source == "mask_sum":
            gate = (mask_sum > self.thresh_count).float()

        elif self.gate_source == "sobel":
            with torch.no_grad():
                xm = (x * mask).mean(dim=1, keepdim=True)
                gx = torch.nn.functional.conv2d(xm, self.sobel_x, padding=1)
                gy = torch.nn.functional.conv2d(xm, self.sobel_y, padding=1)
                grad_mag = (gx * gx + gy * gy).clamp_min(0).sqrt()
                if self.stride_val > 1:
                    grad_mag = torch.nn.functional.avg_pool2d(grad_mag, self.stride_val)
                grad_mag = grad_mag * ref_active
                gate = self._coverage_matched_gate(grad_mag, ref_active)

        elif self.gate_source == "energy":
            with torch.no_grad():
                e_map = (x * mask).pow(2).mean(dim=1, keepdim=True)
                e_map = torch.nn.functional.avg_pool2d(e_map, 3, stride=1, padding=1)
                if self.stride_val > 1:
                    e_map = torch.nn.functional.avg_pool2d(e_map, self.stride_val)
                e_map = e_map * ref_active
                gate = self._coverage_matched_gate(e_map, ref_active)

        elif self.gate_source == "random":
            with torch.no_grad():
                rand = torch.rand_like(mask_sum) * ref_active
                gate = self._coverage_matched_gate(rand, ref_active)

        else:
            raise ValueError(f"Unknown DACA gate_source: {self.gate_source}")

        out = out * gate
        return out, gate



class SparseBasicBlock(nn.Module):
    """BasicBlock with sparse convolution and mask propagation."""

    def __init__(self, sparse_conv_cls, inplanes, planes, norm_fn=None):
        super().__init__()
        bias = norm_fn is not None
        self.conv1 = sparse_conv_cls(inplanes, planes, 3,
                                     stride=1, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = sparse_conv_cls(planes, planes, 3,
                                     stride=1, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x, mask):
        identity = x

        out, mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, mask = self.conv2(out, mask)
        out = self.bn2(out)

        out = out + identity
        out = self.relu2(out)
        return out, mask


class SparseStage(nn.Module):
    """
    A backbone stage using sparse convolutions with mask propagation.

    Consists of an initial sparse conv (with optional stride for downsampling)
    followed by one or more SparseBasicBlocks.
    """

    def __init__(self, sparse_conv_cls, in_channels, out_channels,
                 kernel_size=3, stride=1, num_blocks=1, norm_fn=None):
        super().__init__()
        padding = kernel_size // 2

        # Initial conv (may have stride > 1 for downsampling)
        self.init_conv = sparse_conv_cls(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.init_bn = norm_fn(out_channels)
        self.init_relu = nn.ReLU()

        # Basic blocks (stride=1, same in/out channels)
        self.blocks = nn.ModuleList([
            SparseBasicBlock(sparse_conv_cls, out_channels, out_channels,
                             norm_fn=norm_fn)
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask):
        x, mask = self.init_conv(x, mask)
        x = self.init_bn(x)
        x = self.init_relu(x)

        for block in self.blocks:
            x, mask = block(x, mask)

        return x, mask


# ============================================================
# Native spconv stages for apple-to-apple comparison
# ============================================================

class SpconvBasicBlock(nn.Module):
    """BasicBlock using native spconv SubMConv2d. Mirrors SparseBasicBlock."""

    def __init__(self, inplanes, planes, norm_fn=None, indice_key=None):
        super().__init__()
        import spconv.pytorch as spconv
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, 3, padding=1, bias=bias, indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, 3, padding=1, bias=bias, indice_key=indice_key)
        self.bn2 = norm_fn(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu2(out.features))
        return out


class SpconvStage(nn.Module):
    """
    Native spconv stage mirroring SparseStage structure:
    init_conv (with optional stride) + N SpconvBasicBlocks.
    Each stage gets unique indice_keys to avoid index table conflicts.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 num_blocks=1, norm_fn=None, stage_id=1):
        super().__init__()
        import spconv.pytorch as spconv
        padding = kernel_size // 2

        if stride == 1:
            self.init_conv = spconv.SubMConv2d(
                in_channels, out_channels, kernel_size, padding=padding,
                bias=False, indice_key=f'subm_init_s{stage_id}')
        else:
            self.init_conv = spconv.SparseConv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, bias=False, indice_key=f'spconv_s{stage_id}')
        self.init_bn = norm_fn(out_channels)
        self.init_relu = nn.ReLU()

        self.blocks = nn.ModuleList([
            SpconvBasicBlock(out_channels, out_channels, norm_fn=norm_fn,
                             indice_key=f'res_s{stage_id}')
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.init_conv(x)
        x = x.replace_feature(self.init_bn(x.features))
        x = x.replace_feature(self.init_relu(x.features))

        for block in self.blocks:
            x = block(x)
        return x
