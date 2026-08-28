#!/usr/bin/env python3
"""Generate Conv2d-VFE train BEV for 619 step7 paired indices.

Run in any env with working torch + pcdet (your normal training env).
Output: <output_dir>/bev_{train_idx:05d}.bin -- one per index in indices file.
"""
import os, sys, time, argparse, numpy as np, torch
from torch.utils.data import Subset

parser = argparse.ArgumentParser()
parser.add_argument('--pcdet_root', default=os.path.expanduser('~/WORKDIR'),
                    help='pcdet source root (contains pcdet/, tools/)')
parser.add_argument('--cfg_file',  default=None,
                    help='default: <pcdet_root>/tools/cfgs/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint.yaml')
parser.add_argument('--ckpt',      default=None,
                    help='default: <pcdet_root>/output/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint/v2/ckpt/latest_model.pth')
parser.add_argument('--indices_file', required=True,
                    help='text file with one train index per line (e.g. step7 union 619 indices)')
parser.add_argument('--output_dir', required=True,
                    help='output dir for bev_{idx:05d}.bin files')
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()

if args.cfg_file is None:
    args.cfg_file = os.path.join(args.pcdet_root,
        'tools/cfgs/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint.yaml')
if args.ckpt is None:
    args.ckpt = os.path.join(args.pcdet_root,
        'output/nuscenes_models/cbgs_pillar0075_densepillar_centerpoint/v2/ckpt/latest_model.pth')

sys.path.insert(0, args.pcdet_root)
sys.path.insert(0, os.path.join(args.pcdet_root, 'tools/deploy'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

os.makedirs(args.output_dir, exist_ok=True)

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
model.eval().cuda() if torch.cuda.is_available() else model.eval()

backbone_3d = model.backbone_3d
t0 = time.time()

with torch.no_grad():
    for i, bd in enumerate(loader):
        orig_idx = indices[i]
        if torch.cuda.is_available():
            load_data_to_gpu(bd)
        bd = model.vfe(bd)
        bd['batch_size'] = 1
        spatial = backbone_3d.scatter_pillars_to_bev(1, bd['pillar_features'], bd['pillar_coords'])
        spatial.cpu().numpy().tofile(os.path.join(args.output_dir, f'bev_{orig_idx:05d}.bin'))
        if (i + 1) % 20 == 0:
            sp = (i + 1) / (time.time() - t0)
            print(f'  [{i+1}/{len(indices)}] {sp:.1f}/s ETA {(len(indices)-i-1)/sp:.0f}s', flush=True)

print(f'\nDone: {len(indices)} BEVs in {args.output_dir} ({time.time()-t0:.0f}s)', flush=True)
