import h5py
import numpy as np
from mmdet.datasets.builder import PIPELINES


CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


@PIPELINES.register_module()
class LoadVJepaFeaturesFromH5:
    """
    Loads cached V-JEPA features.

    H5:
        vitb/<sample_token> -> [6, 672, 768]
    """

    def __init__(self, h5_path, group="vitb", img_w=656, img_h=368, orig_w=1600, orig_h=900):
        self.h5_path = h5_path
        self.group = group
        self.img_w = img_w
        self.img_h = img_h
        self.orig_w = orig_w
        self.orig_h = orig_h
        self._h5 = None

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _get_sample_token(self, results):
        if "sample_idx" in results:
            return results["sample_idx"]
        if "token" in results:
            return results["token"]
        raise KeyError(f"Cannot find sample token. Available keys: {list(results.keys())}")

    def _camera_from_path(self, path):
        path = str(path)
        for cam in CAMERAS:
            if cam in path:
                return cam
        return None

    def _reorder_to_metadata_camera_order(self, feat, results):
        """
        Cached features have fixed CAMERAS order.
        BEVFormer metadata/lidar2img has the order of img_filename.
        We reorder features so camera features match camera matrices.
        """
        filenames = results.get("img_filename", None)
        if filenames is None:
            return feat

        meta_order = [self._camera_from_path(p) for p in filenames]
        if any(cam is None for cam in meta_order):
            return feat

        perm = [CAMERAS.index(cam) for cam in meta_order]
        return feat[perm]
    def _scale_geometry_to_vjepa_resolution(self, results):
        sx = self.img_w / self.orig_w
        sy = self.img_h / self.orig_h

        if "lidar2img" in results:
            scaled_lidar2img = []
            for mat in results["lidar2img"]:
                mat = np.array(mat, dtype=np.float32).copy()
                mat[0, :] *= sx
                mat[1, :] *= sy
                scaled_lidar2img.append(mat)
            results["lidar2img"] = scaled_lidar2img

        if "cam_intrinsic" in results:
            scaled_intrinsics = []
            for mat in results["cam_intrinsic"]:
                mat = np.array(mat, dtype=np.float32).copy()
                mat[0, :] *= sx
                mat[1, :] *= sy
                scaled_intrinsics.append(mat)
            results["cam_intrinsic"] = scaled_intrinsics

        return results

    def __call__(self, results):
        token = self._get_sample_token(results)
        key = f"{self.group}/{token}"

        h5 = self._get_h5()
        if key not in h5:
            raise KeyError(f"Missing V-JEPA feature in h5: {key}")

        feat = np.asarray(h5[key][:])  # [6, 672, 768], float16

        if feat.ndim != 3:
            raise ValueError(f"Expected [6, tokens, dim], got {feat.shape}")
        if feat.shape[0] != 6:
            raise ValueError(f"Expected 6 cameras, got {feat.shape[0]}")

        feat = self._reorder_to_metadata_camera_order(feat, results)

        results = self._scale_geometry_to_vjepa_resolution(results)

        # Important: BEVFormer receives this as img=...
        results["img"] = feat

        # Keep original nuScenes geometry metadata.
        num_cams = feat.shape[0]
        results["img_shape"] = [(self.img_h, self.img_w, 3) for _ in range(num_cams)]
        results["ori_shape"] = [(self.img_h, self.img_w, 3) for _ in range(num_cams)]
        results["pad_shape"] = [(self.img_h, self.img_w, 3) for _ in range(num_cams)]

        return results