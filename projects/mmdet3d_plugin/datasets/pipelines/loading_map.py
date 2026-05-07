import pickle
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES


# Maps each logical class name to the nuScenes map-layer names that compose it.
# 'divider' merges road_divider and lane_divider (they don't exist as a single
# layer in the nuScenes API).
_MAP_CLASS_TO_LAYERS = {
    'drivable_area': ['drivable_area'],
    'ped_crossing':  ['ped_crossing'],
    'walkway':       ['walkway'],
    'stop_line':     ['stop_line'],
    'carpark_area':  ['carpark_area'],
    'divider':       ['road_divider', 'lane_divider'],
}


@PIPELINES.register_module()
class LoadMapMask:
    """Rasterise nuScenes HD-map layers onto the BEV grid.

    Produces ``results['gt_masks_bev']`` — a float32 tensor of shape
    ``(num_classes, bev_h, bev_w)`` with binary (0/1) values per class.

    The BEV grid orientation matches BEVFormer's convention:
        - rows  → Y axis (ego left-right,  index 0 = right / Y_min)
        - cols  → X axis (ego forward-back, index 0 = back  / X_min)

    Args:
        data_root (str): Path to the nuScenes dataset root.
        xbound (list[float]): [x_min, x_max, dx] in metres. E.g. [-51.2, 51.2, 0.512].
        ybound (list[float]): [y_min, y_max, dy] in metres. E.g. [-51.2, 51.2, 0.512].
        classes (list[str]): Ordered list of map class names drawn from
            _MAP_CLASS_TO_LAYERS keys. Default: all 6 nuScenes map classes.
        version (str): nuScenes version string used to initialise the NuScenes
            object for building the sample-token → location lookup table.
    """

    def __init__(
        self,
        data_root,
        xbound,
        ybound,
        classes=('drivable_area', 'ped_crossing', 'walkway',
                 'stop_line', 'carpark_area', 'divider'),
        version='v1.0-trainval',
    ):
        self.data_root = data_root
        self.classes = list(classes)
        self.xbound = xbound  # [xmin, xmax, dx]
        self.ybound = ybound  # [ymin, ymax, dy]

        # BEV canvas size (pixels)
        self.canvas_h = round((ybound[1] - ybound[0]) / ybound[2])  # rows = Y
        self.canvas_w = round((xbound[1] - xbound[0]) / xbound[2])  # cols = X

        # Map patch extent in metres
        self.patch_h = ybound[1] - ybound[0]  # metres along Y
        self.patch_w = xbound[1] - xbound[0]  # metres along X

        # Collect all unique nuScenes layer names needed across all classes
        self._all_layers = []
        seen = set()
        for cls in self.classes:
            for layer in _MAP_CLASS_TO_LAYERS.get(cls, [cls]):
                if layer not in seen:
                    self._all_layers.append(layer)
                    seen.add(layer)
        self._layer_idx = {name: i for i, name in enumerate(self._all_layers)}

        # Build sample-token → map-location mapping.
        # NuScenes is initialised once here (main process); worker processes
        # receive a pickled copy via the DataLoader fork.
        self._token2location = self._build_token_location_map(data_root, version)

        # Cache one NuScenesMap instance per location.
        from nuscenes.map_expansion.map_api import NuScenesMap
        locations = set(self._token2location.values())
        self._maps = {
            loc: NuScenesMap(dataroot=data_root, map_name=loc)
            for loc in locations
        }

    @staticmethod
    def _build_token_location_map(data_root, version):
        from nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
        token2loc = {}
        for sample in nusc.sample:
            scene = nusc.get('scene', sample['scene_token'])
            log   = nusc.get('log',   scene['log_token'])
            token2loc[sample['token']] = log['location']
        return token2loc

    def __call__(self, results):
        sample_token    = results['sample_idx']
        ego_translation = results['ego2global_translation']  # [x, y, z]
        ego_rotation    = results['ego2global_rotation']     # quaternion [w, x, y, z]

        location = self._token2location.get(sample_token)
        if location is None:
            # Fallback for tokens not in the lookup (e.g. test split without map)
            gt_masks = np.zeros(
                (len(self.classes), self.canvas_h, self.canvas_w), dtype=np.float32)
            results['gt_masks_bev'] = DC(
                torch.from_numpy(gt_masks), cpu_only=False, stack=True)
            return results

        from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

        nusc_map    = self._maps[location]
        map_center  = (float(ego_translation[0]), float(ego_translation[1]))
        patch_angle = quaternion_yaw(Quaternion(ego_rotation)) / np.pi * 180

        # patch_box: (x_center, y_center, height_in_Y, width_in_X) [metres]
        patch_box = (map_center[0], map_center[1], self.patch_h, self.patch_w)

        # canvas shape: (num_layers, canvas_h, canvas_w)
        # canvas_h rows span Y direction, canvas_w cols span X direction.
        canvas = nusc_map.get_map_mask(
            patch_box, patch_angle, self._all_layers,
            canvas_size=(self.canvas_h, self.canvas_w),
        )

        # nuScenes renders row-0 at Y_max (ego-left).
        # BEVFormer BEV row-0 = Y_min (ego-right).  Flip rows to align.
        canvas = canvas[:, ::-1, :].copy()

        # Combine nuScenes layers into the logical class masks.
        gt_masks = np.zeros(
            (len(self.classes), self.canvas_h, self.canvas_w), dtype=np.float32)
        for cls_idx, cls_name in enumerate(self.classes):
            for layer_name in _MAP_CLASS_TO_LAYERS.get(cls_name, [cls_name]):
                if layer_name in self._layer_idx:
                    gt_masks[cls_idx] = np.logical_or(
                        gt_masks[cls_idx], canvas[self._layer_idx[layer_name]])

        results['gt_masks_bev'] = DC(
            torch.from_numpy(gt_masks), cpu_only=False, stack=True)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'classes={self.classes}, '
                f'canvas=({self.canvas_h},{self.canvas_w}), '
                f'patch=({self.patch_h}m x {self.patch_w}m))')


@PIPELINES.register_module()
class LoadMapMaskFromPkl:
    """Load pre-rasterised HD-map masks from a pickle file.

    Replaces the on-the-fly ``LoadMapMask`` rasteriser with a simple lookup
    into a pre-computed dict, matching the pattern used by
    ``LoadVJepaFeaturesFromH5`` for V-JEPA cached features.

    Expected pkl format (produced by the offline extraction script)::

        {
            "<sample_token>": np.ndarray,  # shape (num_classes, H, W), uint8/bool
            ...
        }

    The masks are binary: 1 = map class present, 0 = absent.

    Args:
        mask_pkl_path (str): Path to the pickle file.
        classes (tuple[str]): Ordered map class names.  Must match the order
            used when the pkl was generated.
        num_classes (int): Number of map classes (used for the zero fallback).
    """

    # Default class order matching nuScenes + BEVFusion convention
    _DEFAULT_CLASSES = (
        'drivable_area', 'ped_crossing', 'walkway',
        'stop_line', 'carpark_area', 'divider',
    )

    def __init__(
        self,
        mask_pkl_path,
        classes=None,
        num_classes=6,
    ):
        self.mask_pkl_path = mask_pkl_path
        self.classes = list(classes) if classes is not None else list(self._DEFAULT_CLASSES)
        self.num_classes = num_classes if classes is None else len(self.classes)

        # Lazy-load: opened once per worker process, never in the main process.
        # This mirrors the h5py lazy-open pattern in LoadVJepaFeaturesFromH5.
        self._masks = None
        self._mask_shape = None   # inferred on first successful lookup

    def _load(self):
        with open(self.mask_pkl_path, 'rb') as f:
            self._masks = pickle.load(f)

    def __call__(self, results):
        if self._masks is None:
            self._load()

        token = results['sample_idx']
        raw = self._masks.get(token, None)

        if raw is None:
            # Token missing (e.g. test split has no map GT).
            # Use the shape of the first seen mask, or fall back to num_classes.
            if self._mask_shape is not None:
                C, H, W = self._mask_shape
            else:
                C, H, W = self.num_classes, 200, 200
            mask = np.zeros((C, H, W), dtype=np.float32)
        else:
            mask = np.asarray(raw, dtype=np.float32)
            if mask.ndim != 3:
                raise ValueError(
                    f'Expected mask shape (C, H, W) for token {token}, '
                    f'got {mask.shape}')
            if self._mask_shape is None:
                self._mask_shape = mask.shape

        results['gt_masks_bev'] = DC(
            torch.from_numpy(mask), cpu_only=False, stack=True)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'mask_pkl_path={self.mask_pkl_path}, '
                f'classes={self.classes})')
