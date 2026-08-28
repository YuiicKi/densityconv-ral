# Aggregate activation statistics over the nuScenes val set.
# Hooks every Conv2d / BatchNorm2d / ReLU output in the model, plus the
# DensityConv occupancy-ratio branch (mask_conv output / K^2), and accumulates:
#   - exact global |max|, per-sample |max| (for cross-frame variability),
#   - a dynamically-ranged histogram -> p50/p99/p99.9/p99.99/p99.999,
#   - zero fraction, mean/std of |x|.
# Outputs: act_stats_<tag>.json (summary) + act_stats_<tag>.npz (histograms).
import argparse
import json
import os

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

BINS = 2048


class HistObserver:
    def __init__(self, kind):
        self.kind = kind          # 'conv' | 'bn' | 'relu' | 'density_scale'
        self.range = 1.0 if kind == 'density_scale' else None
        self.hist = torch.zeros(BINS, dtype=torch.float64)
        self.exact_max = 0.0
        self.per_sample_max = []
        self.n = 0
        self.zeros = 0
        self.s = 0.0
        self.ss = 0.0

    @torch.no_grad()
    def update(self, t):
        a = t.detach().abs().float()
        smax = a.amax(dim=tuple(range(1, a.dim()))) if a.dim() > 1 else a.amax().unsqueeze(0)
        self.per_sample_max.extend(smax.cpu().tolist())
        m = float(smax.max().item())
        self.exact_max = max(self.exact_max, m)
        flat = a.flatten()
        self.n += flat.numel()
        self.zeros += int((flat == 0).sum().item())
        self.s += float(flat.sum().item())
        self.ss += float((flat * flat).sum().item())
        if self.range is None:
            self.range = max(m, 1e-12)
        while m > self.range:  # double range, fold existing bins pairwise
            self.hist = torch.cat([self.hist.reshape(-1, 2).sum(1),
                                   torch.zeros(BINS // 2, dtype=torch.float64)])
            self.range *= 2
        self.hist += torch.histc(flat, bins=BINS, min=0.0, max=self.range).double().cpu()

    def percentile(self, q):
        c = torch.cumsum(self.hist, 0)
        total = c[-1].item()
        if total == 0:
            return 0.0
        idx = int(torch.searchsorted(c, torch.tensor(q / 100.0 * total)).item())
        return (min(idx, BINS - 1) + 1) / BINS * self.range

    def summary(self):
        mean = self.s / max(self.n, 1)
        var = max(self.ss / max(self.n, 1) - mean * mean, 0.0)
        psm = np.array(self.per_sample_max)
        return {
            'kind': self.kind, 'n': self.n, 'exact_max': self.exact_max,
            'p50': self.percentile(50), 'p99': self.percentile(99),
            'p99_9': self.percentile(99.9), 'p99_99': self.percentile(99.99),
            'p99_999': self.percentile(99.999),
            'zero_frac': self.zeros / max(self.n, 1),
            'abs_mean': mean, 'abs_std': var ** 0.5,
            'sample_max_mean': float(psm.mean()) if len(psm) else 0.0,
            'sample_max_std': float(psm.std()) if len(psm) else 0.0,
            'sample_max_min': float(psm.min()) if len(psm) else 0.0,
            'sample_max_max': float(psm.max()) if len(psm) else 0.0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_frames', type=int, default=0, help='0 = full val')
    parser.add_argument('--out_dir', default=os.path.expanduser('~/WORKDIR'))
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size, dist=False, workers=4,
        logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()

    observers, hooks = {}, []
    density_parents = {}  # mask_conv module -> parent k_sq
    for name, mod in model.named_modules():
        if hasattr(mod, 'mask_conv') and hasattr(mod, 'k_sq'):
            density_parents[mod.mask_conv] = (name, float(mod.k_sq.item()))
    skip = set(density_parents.keys())

    def add_hook(name, mod, kind):
        obs = observers.setdefault(name, HistObserver(kind))
        def fn(m, i, o, _obs=obs):
            out = o[0] if isinstance(o, tuple) else o
            _obs.update(out)
        hooks.append(mod.register_forward_hook(fn))

    for name, mod in model.named_modules():
        if mod in skip:
            pname, k_sq = density_parents[mod]
            obs = observers.setdefault(pname + '.density_scale', HistObserver('density_scale'))
            def fn(m, i, o, _obs=obs, _k=k_sq):
                _obs.update(o.detach() / _k)
            hooks.append(mod.register_forward_hook(fn))
        elif isinstance(mod, torch.nn.Conv2d):
            add_hook(name, mod, 'conv')
        elif isinstance(mod, torch.nn.BatchNorm2d):
            add_hook(name, mod, 'bn')
        elif isinstance(mod, torch.nn.ReLU):
            add_hook(name, mod, 'relu')

    logger.info('hooked %d modules (%d density branches)' % (len(observers), len(density_parents)))

    done = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            load_data_to_gpu(batch)
            model(batch)
            done += batch['batch_size']
            if i % 100 == 0:
                logger.info('frames %d' % done)
            if args.max_frames and done >= args.max_frames:
                break
    for h in hooks:
        h.remove()

    summary = {n: o.summary() for n, o in observers.items()}
    jpath = os.path.join(args.out_dir, 'act_stats_%s.json' % args.tag)
    with open(jpath, 'w') as f:
        json.dump({'tag': args.tag, 'ckpt': args.ckpt, 'frames': done, 'layers': summary}, f, indent=1)
    np.savez_compressed(
        os.path.join(args.out_dir, 'act_stats_%s.npz' % args.tag),
        **{n + '/hist': o.hist.numpy() for n, o in observers.items()},
        **{n + '/range': np.array([o.range or 0.0]) for n, o in observers.items()},
        **{n + '/sample_max': np.array(o.per_sample_max) for n, o in observers.items()})
    logger.info('saved %s (+.npz), frames=%d' % (jpath, done))


if __name__ == '__main__':
    main()
