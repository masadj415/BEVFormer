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

        # Let the normal 3D formatter handle gt_bboxes_3d, gt_labels_3d, etc.
        results = super().__call__(results)

        if not torch.is_tensor(img):
            img = torch.from_numpy(img)

        results["img"] = DC(img, stack=True)

        return results