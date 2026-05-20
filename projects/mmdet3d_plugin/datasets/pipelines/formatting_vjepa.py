import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D


@PIPELINES.register_module()
class VJepaFormatBundle3D(DefaultFormatBundle3D):
    """
    Format V-JEPA cached features without treating them as RGB images.

    Input:
        results["img"]: numpy array [6, 672, 768]

    Output:
        results["img"]: DataContainer(torch.Tensor [6, 672, 768])
    """

    def __call__(self, results):
        img = results.pop("img")

        # Stash gt_future_ego before the parent formatter sees it (parent
        # doesn't know this key and would leave it as a raw numpy array).
        gt_future_ego = results.pop("gt_future_ego", None)

        # Let the normal 3D formatter handle gt_bboxes_3d, gt_labels_3d, etc.
        results = super().__call__(results)

        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        results["img"] = DC(img, stack=True)

        if gt_future_ego is not None:
            if not torch.is_tensor(gt_future_ego):
                gt_future_ego = torch.from_numpy(gt_future_ego)
            # pad_dims=None: tensor is [6,2] (ndim=2), default pad_dims=2 would
            # trigger assert ndim > pad_dims (2>2 is False).
            results["gt_future_ego"] = DC(gt_future_ego, stack=True, cpu_only=False, pad_dims=None)

        # Per-agent future trajectories from the motion pkl.
        # N varies per sample → stack=False so the collate gives a list of tensors.
        for key in ("gt_fut_traj", "gt_fut_traj_mask"):
            val = results.pop(key, None)
            if val is not None:
                if not torch.is_tensor(val):
                    val = torch.from_numpy(val.astype("float32"))
                results[key] = DC(val, stack=False, cpu_only=False, pad_dims=None)

        return results