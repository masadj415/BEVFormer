import mmcv
import numpy as np
import os
import math
import time
import argparse
from os import path as osp
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm

# All 10 rasterizable layers from the nuScenes map expansion pack.
# 'traffic_light' is the 11th but stored as point nodes — not rasterizable via get_map_mask().
SEG_LAYERS = [
    'drivable_area',   # 0  drivable road surface
    'road_segment',    # 1  road segment polygon
    'road_block',      # 2  road block (intersection interior)
    'lane',            # 3  individual lane polygon
    'ped_crossing',    # 4  pedestrian crossing
    'walkway',         # 5  sidewalk / footpath
    'stop_line',       # 6  stop line at intersection
    'carpark_area',    # 7  car park polygon
    'road_divider',    # 8  road center divider line
    'lane_divider',    # 9  lane boundary line
]

MAPS = [
    'boston-seaport',
    # singapore maps not downloaded
    # 'singapore-hollandvillage',
    # 'singapore-onenorth',
    # 'singapore-queenstown',
]


def obtain_map_info_v12(nusc_maps, location, l2e_r_mat, l2e_t, e2g_r_mat, e2g_t,
                        lidar_path, map_out_dir,
                        patch_size=(102.4, 102.4), canvas_size=(200, 200)):
    """Extract HD map mask for one sample and save as .npz.

    Returns:
        dict with key 'map_mask' pointing to the saved .npz path.
        The .npz contains array 'arr_0' of shape (len(SEG_LAYERS), H, W),
        dtype uint8, one binary channel per SEG_LAYER, in LiDAR frame orientation.
    """
    nusc_map = nusc_maps[location]

    # lidar -> global transform
    l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
    l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

    patch_box = (l2g_t[0], l2g_t[1], patch_size[0], patch_size[1])
    patch_angle = math.degrees(Quaternion(matrix=l2g_r_mat).yaw_pitch_roll[0])

    # shape: (num_layers, canvas_H, canvas_W)
    map_mask = nusc_map.get_map_mask(patch_box, patch_angle, SEG_LAYERS,
                                     canvas_size=canvas_size)
    # rotate 180° to align map frame with LiDAR frame (forward = up)
    map_mask = np.rot90(map_mask, k=2, axes=(1, 2)).astype(np.uint8)

    fname = osp.splitext(osp.basename(lidar_path))[0] + '.npz'
    save_path = osp.join(map_out_dir, fname)
    np.savez_compressed(save_path, map_mask)
    return {'map_mask': save_path}


def add_map_to_pkl(pkl_path, nusc_maps, location, map_out_dir, out_pkl_path, split_name,
                   canvas_size=(200, 200)):
    """Load an existing BEVFormer pkl, add 'maps' to every sample, re-save.

    Skips samples whose .npz already exists so the job is safely resumable.
    """
    print(f'\n{"="*60}')
    print(f'  Loading pkl: {pkl_path}')
    data = mmcv.load(pkl_path)
    infos = data['infos']
    total = len(infos)
    print(f'  Samples to process: {total}')
    print(f'  Output map dir:     {map_out_dir}')
    print(f'  Output pkl:         {out_pkl_path}')
    print(f'{"="*60}')

    already_done = 0
    errors = 0
    t_start = time.time()

    pbar = tqdm(infos, desc=split_name, unit='sample',
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for i, info in enumerate(pbar):
        lidar_path = str(info['lidar_path'])
        fname = osp.splitext(osp.basename(lidar_path))[0] + '.npz'
        save_path = osp.join(map_out_dir, fname)

        # resume: skip if already extracted
        if osp.exists(save_path):
            info['maps'] = {'map_mask': save_path}
            already_done += 1
            pbar.set_postfix(skipped=already_done, errors=errors, refresh=False)
            continue

        try:
            l2e_r_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            l2e_t = np.array(info['lidar2ego_translation'])
            e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix
            e2g_t = np.array(info['ego2global_translation'])

            info['maps'] = obtain_map_info_v12(
                nusc_maps, location,
                l2e_r_mat, l2e_t, e2g_r_mat, e2g_t,
                lidar_path, map_out_dir,
                canvas_size=canvas_size,
            )
        except Exception as e:
            errors += 1
            tqdm.write(f'[ERROR] sample {i} token={info["token"]}: {e}')

        pbar.set_postfix(skipped=already_done, errors=errors, refresh=False)

        # print a status line every 500 samples so tmux log is readable
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1 - already_done) / max(elapsed, 1)
            remaining = (total - i - 1) / max(rate, 1e-6)
            tqdm.write(
                f'[{split_name}] {i+1}/{total} done | '
                f'skipped={already_done} errors={errors} | '
                f'rate={rate:.1f} s/sample | '
                f'ETA {remaining/60:.1f} min'
            )

    pbar.close()

    elapsed_total = time.time() - t_start
    processed = total - already_done
    print(f'\n  Finished {split_name}: {total} samples in {elapsed_total/60:.1f} min')
    print(f'  Newly extracted: {processed} | Skipped (already existed): {already_done} | Errors: {errors}')

    print(f'  Saving augmented pkl ...', end=' ', flush=True)
    mmcv.dump(data, out_pkl_path)
    print(f'done.')
    print(f'  -> {out_pkl_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Augment existing BEVFormer pkl files with nuScenes v1.2 HD map masks.'
    )
    parser.add_argument('--root-path', type=str, required=True,
                        help='Path to nuScenes dataset root (must contain maps/expansion/*.json)')
    parser.add_argument('--train-pkl', type=str, default=None,
                        help='Path to existing BEVFormer train pkl')
    parser.add_argument('--val-pkl', type=str, default=None,
                        help='Path to existing BEVFormer val pkl')
    parser.add_argument('--map-out-dir', type=str, required=True,
                        help='Directory where per-sample .npz map masks will be written')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Directory where augmented pkl files will be written')
    parser.add_argument('--location', type=str, default='boston-seaport',
                        choices=MAPS,
                        help='Map location to use for all samples (default: boston-seaport)')
    parser.add_argument('--canvas-size', type=int, default=200,
                        help='BEV grid size: 200 for base, 150 for small, 50 for tiny (default: 200)')
    args = parser.parse_args()

    if args.train_pkl is None and args.val_pkl is None:
        raise ValueError('Provide at least one of --train-pkl or --val-pkl')

    os.makedirs(args.map_out_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'\nUsing map location: {args.location}')
    print(f'Loading HD maps for {len(MAPS)} locations ...')
    nusc_maps = {}
    for loc in MAPS:
        nusc_maps[loc] = NuScenesMap(dataroot=args.root_path, map_name=loc)
        print(f'  loaded: {loc}')

    canvas_size = (args.canvas_size, args.canvas_size)
    res = 102.4 / args.canvas_size
    print(f'\nCanvas size : {args.canvas_size}x{args.canvas_size} px  ({res:.4f} m/px)')
    print(f'Patch size  : 102.4m x 102.4m  (matches BEVFormer point_cloud_range ±51.2m)')
    print(f'Layers to extract ({len(SEG_LAYERS)}):')
    for i, name in enumerate(SEG_LAYERS):
        print(f'  [{i}] {name}')

    t0 = time.time()
    suffix = f'_with_map_{args.canvas_size}.pkl'

    if args.train_pkl:
        out_name = osp.basename(args.train_pkl).replace('.pkl', suffix)
        add_map_to_pkl(args.train_pkl, nusc_maps, args.location, args.map_out_dir,
                       osp.join(args.out_dir, out_name), split_name='TRAIN',
                       canvas_size=canvas_size)

    if args.val_pkl:
        out_name = osp.basename(args.val_pkl).replace('.pkl', suffix)
        add_map_to_pkl(args.val_pkl, nusc_maps, args.location, args.map_out_dir,
                       osp.join(args.out_dir, out_name), split_name='VAL',
                       canvas_size=canvas_size)

    total_min = (time.time() - t0) / 60
    print(f'\nAll done in {total_min:.1f} min.')
    print(f'Map masks (.npz) : {args.map_out_dir}')
    print(f'Augmented pkls   : {args.out_dir}')


LAYER_COLORS = [
    (0.20, 0.60, 0.86),   # 0 drivable_area    — blue
    (0.13, 0.70, 0.67),   # 1 road_segment      — teal
    (0.18, 0.80, 0.44),   # 2 road_block        — green
    (0.95, 0.77, 0.06),   # 3 lane              — yellow
    (0.96, 0.51, 0.19),   # 4 ped_crossing      — orange
    (0.91, 0.30, 0.24),   # 5 walkway           — red
    (0.61, 0.15, 0.69),   # 6 stop_line         — purple
    (0.17, 0.24, 0.31),   # 7 carpark_area      — dark grey
    (1.00, 0.41, 0.71),   # 8 road_divider      — pink
    (0.00, 1.00, 1.00),   # 9 lane_divider      — cyan
]


def make_overlay(mask):
    """Blend all 10 layers into one RGB image with per-layer colors.
    Layers rendered back-to-front so smaller/finer features appear on top.
    """
    H, W = mask.shape[1], mask.shape[2]
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    # render in order: large areas first, fine lines last
    render_order = [7, 2, 1, 0, 5, 4, 3, 6, 8, 9]
    for i in render_order:
        color = np.array(LAYER_COLORS[i], dtype=np.float32)
        where = mask[i].astype(bool)
        canvas[where] = color
    return canvas


def verify(map_out_dir, n_samples=5, out_png='/tmp/map_verify.png'):
    """Load n random .npz files, print per-layer stats, and save:
      - left panel:  composite overlay (all layers, different colors)
      - right panel: 2x5 grid of individual binary masks
    """
    import glob, random
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    files = sorted(glob.glob(osp.join(map_out_dir, '*.npz')))
    if not files:
        print(f'No .npz files found in {map_out_dir}')
        return

    print(f'\nFound {len(files)} .npz files in {map_out_dir}')
    samples = random.sample(files, min(n_samples, len(files)))

    for path in samples:
        mask = np.load(path)['arr_0']
        print(f'\n{osp.basename(path)}')
        print(f'  shape={mask.shape}  dtype={mask.dtype}  values={np.unique(mask).tolist()}')
        for i, name in enumerate(SEG_LAYERS):
            pct = mask[i].sum() / (mask.shape[1] * mask.shape[2]) * 100
            bar = '#' * int(pct / 2)
            print(f'  [{i}] {name:<20s}  {mask[i].sum():5d} px  ({pct:5.1f}%)  {bar}')

    mask = np.load(samples[0])['arr_0']
    H, W = mask.shape[1], mask.shape[2]
    overlay = make_overlay(mask)

    fig = plt.figure(figsize=(22, 9))
    # --- left: composite overlay ---
    ax_ov = fig.add_axes([0.01, 0.05, 0.35, 0.88])
    ax_ov.imshow(overlay, origin='upper')
    ax_ov.plot(W // 2, H // 2, 'w+', markersize=14, markeredgewidth=2.5)
    ax_ov.set_title('Composite overlay (ego = white +)', fontsize=11)
    ax_ov.axis('off')
    legend_handles = [
        Patch(facecolor=LAYER_COLORS[i], label=f'[{i}] {SEG_LAYERS[i]}')
        for i in range(len(SEG_LAYERS))
    ]
    ax_ov.legend(handles=legend_handles, loc='lower left',
                 fontsize=7, framealpha=0.8, ncol=1)

    # --- right: 2x5 individual masks ---
    for i in range(10):
        row, col = divmod(i, 5)
        left = 0.38 + col * 0.123
        bottom = 0.54 - row * 0.46
        ax = fig.add_axes([left, bottom, 0.115, 0.42])
        colored = np.zeros((H, W, 3), dtype=np.float32)
        colored[mask[i].astype(bool)] = LAYER_COLORS[i]
        ax.imshow(colored, origin='upper')
        ax.plot(W // 2, H // 2, 'w+', markersize=8, markeredgewidth=1.5)
        ax.set_title(f'[{i}] {SEG_LAYERS[i]}', fontsize=8)
        ax.axis('off')

    fig.suptitle(
        f'HD Map — {H}x{W} px @ 0.512 m/px  |  102.4m x 102.4m patch\n'
        f'{osp.basename(samples[0])}',
        fontsize=10, y=0.99)

    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    print(f'\nVisualization saved -> {out_png}')


if __name__ == '__main__':
    import sys
    if '--verify' in sys.argv:
        # quick verify mode: python create_HD_map.py --verify --map-out-dir <dir> [--out-png /tmp/x.png]
        import argparse as _ap
        p = _ap.ArgumentParser()
        p.add_argument('--verify', action='store_true')
        p.add_argument('--map-out-dir', type=str, required=True)
        p.add_argument('--n-files', type=int, default=5)
        p.add_argument('--out-png', type=str, default='/tmp/map_verify.png')
        a = p.parse_args()
        verify(a.map_out_dir, n_samples=a.n_files, out_png=a.out_png)
    else:
        main()
