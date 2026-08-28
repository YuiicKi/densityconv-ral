#!/usr/bin/env python3
"""Generate DC-VFE (DensityConv ckpt) train BEV + mask for given indices.

Rebuild of the paper's random-256 calibration sets (expD/E/F indices).
CPU/GPU both work (no GPU => CPU). Output:
  <output_dir>/bev/bev_{train_idx:05d}.bin
  <output_dir>/mask/mask_{train_idx:05d}.bin
"""
import os, sys, time, argparse, numpy as np, torch
from torch.utils.data import Subset

parser = argparse.ArgumentParser()
parser.add_argument('--pcdet_root', default=os.path.expanduser('~/WORKDIR'),
                    help='pcdet source root (contains pcdet/, tools/)')
parser.add_argument('--cfg_file',  default=None,
                    help='default: <pcdet_root>/tools/cfgs/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint_densityconv.yaml')
parser.add_argument('--ckpt',      default=None,
                    help='default: <pcdet_root>/output/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint_densityconv/v1/ckpt/latest_model.pth')
parser.add_argument('--indices_file', required=True,
                    help='text file with one train index per line (e.g. expD_indices.txt)')
parser.add_argument('--output_dir', required=True,
                    help='output dir; bev/ and mask/ subdirs are created inside')
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()

if args.cfg_file is None:
    args.cfg_file = os.path.join(args.pcdet_root,
        'tools/cfgs/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint_densityconv.yaml')
if args.ckpt is None:
    args.ckpt = os.path.join(args.pcdet_root,
        'output/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint_densityconv/v1/ckpt/latest_model.pth')

sys.path.insert(0, args.pcdet_root)
sys.path.insert(0, os.path.join(args.pcdet_root, 'tools/deploy'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

# CPU-only fallback: pcdet internals hardcode .cuda() (e.g. dynamic_pillar_vfe grid_size);
# make all .cuda() calls no-ops when no GPU is visible.
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self

bev_dir = os.path.join(args.output_dir, 'bev')
mask_dir = os.path.join(args.output_dir, 'mask')
os.makedirs(bev_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

cfg_from_yaml_file(args.cfg_file, cfg)
# Override INFO_PATH.test -> train.pkl so training=False loads TRAIN data
cfg.DATA_CONFIG.INFO_PATH['test'] = ['nuscenes_infos_10sweeps_train.pkl']

with open(args.indices_file) as f:
    indices = sorted({int(x.strip()) for x in f if x.strip()})
print(f'Need {len(indices)} unique train indices, range [{indices[0]}, {indices[-1]}]', flush=True)

logger = common_utils.create_logger()
full_set, _, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
    batch_size=1, dist=False, workers=0, logger=logger, training=False
)
print(f'Train set loaded: {len(full_set)} samples', flush=True)
assert indices[-1] < len(full_set), f'max idx {indices[-1]} >= train set size {len(full_set)}'

subset = Subset(full_set, indices)
loader = torch.utils.data.DataLoader(
    subset, batch_size=1, shuffle=False,
    num_workers=args.workers, pin_memory=True,
    collate_fn=full_set.collate_batch
)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=full_set)
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
model.load_state_dict(ck['model_state'], strict=True)
model = model.eval().cuda() if torch.cuda.is_available() else model.eval()

backbone_3d = model.backbone_3d
t0 = time.time()

with torch.no_grad():
    for i, bd in enumerate(loader):
        orig_idx = indices[i]
        if torch.cuda.is_available():
            load_data_to_gpu(bd)
        else:
            # CPU path: mimic load_data_to_gpu but keep tensors on CPU (numeric arrays only)
            for k, v in list(bd.items()):
                if isinstance(v, np.ndarray) and (np.issubdtype(v.dtype, np.number) or v.dtype == np.bool_):
                    bd[k] = torch.from_numpy(v)
        bd = model.vfe(bd)
        bd['batch_size'] = 1
        spatial = backbone_3d.scatter_pillars_to_bev(1, bd['pillar_features'], bd['pillar_coords'])
        mask = (spatial.abs().sum(dim=1, keepdim=True) > 0).float()
        spatial.cpu().numpy().tofile(os.path.join(bev_dir, f'bev_{orig_idx:05d}.bin'))
        mask.cpu().numpy().tofile(os.path.join(mask_dir, f'mask_{orig_idx:05d}.bin'))
        if (i + 1) % 20 == 0:
            sp = (i + 1) / (time.time() - t0)
            print(f'  [{i+1}/{len(indices)}] {sp:.1f}/s ETA {(len(indices)-i-1)/sp:.0f}s', flush=True)

print(f'\nDone: {len(indices)} BEVs+masks in {args.output_dir} ({time.time()-t0:.0f}s)', flush=True)
