#!/usr/bin/env python3
"""Preprocess nuscenes_infos pkl into a text index for C++ batch eval.

pkl keys: lidar_path, token, sweeps, ref_from_car (4x4), car_from_global (4x4)
sweep keys: lidar_path, transform_matrix (4x4 or None), time_lag
"""
import pickle, sys
import numpy as np

def strip_path(p):
    for prefix in ['samples/', 'sweeps/']:
        idx = p.find(prefix)
        if idx >= 0:
            return p[idx:]
    return p

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <infos.pkl> <output.txt>")
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f:
        infos = pickle.load(f)

    with open(sys.argv[2], 'w') as f:
        f.write(f"{len(infos)}\n")
        for info in infos:
            token = info['token']
            lp = strip_path(info['lidar_path'])

            # lidar→global = inv(car_from_global) @ inv(ref_from_car)
            ref_from_car = np.array(info['ref_from_car'])    # 4x4, lidar←ego
            car_from_global = np.array(info['car_from_global'])  # 4x4, ego←global
            # ref_from_global = ref_from_car @ car_from_global (global→car→lidar)
            # lidar→global = inv(ref_from_global)
            l2g = np.linalg.inv(ref_from_car @ car_from_global)
            l2g_r = l2g[:3, :3].flatten()  # 9
            l2g_t = l2g[:3, 3]             # 3

            # Filter sweeps: skip those with transform_matrix=None (same as keyframe)
            valid_sweeps = []
            for sw in info.get('sweeps', []):
                if sw['transform_matrix'] is not None and sw['lidar_path'] != info['lidar_path']:
                    valid_sweeps.append(sw)

            f.write(f"{token} {lp}\n")
            f.write(' '.join(f'{v:.8f}' for v in list(l2g_r) + list(l2g_t)) + '\n')
            f.write(f"{len(valid_sweeps)}\n")
            for sw in valid_sweeps:
                sp = strip_path(sw['lidar_path'])
                tm = np.array(sw['transform_matrix'])  # 4x4 sensor2lidar
                s2l_r = tm[:3, :3].flatten()
                s2l_t = tm[:3, 3]
                tl = float(sw.get('time_lag', 0.0))
                f.write(f"{sp}\n")
                f.write(' '.join(f'{v:.8f}' for v in list(s2l_r) + list(s2l_t)) + f' {tl:.8f}\n')

    print(f"Wrote {len(infos)} frames to {sys.argv[2]}")

if __name__ == '__main__':
    main()
