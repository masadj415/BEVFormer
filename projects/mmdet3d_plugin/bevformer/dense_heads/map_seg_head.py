import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS
from mmdet.models.builder import build_loss


@HEADS.register_module()
class MapSegHead(BaseModule):
    """Dense BEV map segmentation head.

    Architecture follows BEVFusion (Liu et al., 2023): a lightweight
    convolutional decoder on the BEV feature map with independent per-class
    sigmoid predictions and focal loss.

    The BEV encoder (200×200×256) already runs at the target output resolution,
    so no upsampling is needed — the head is purely classification.

    Each map class is an independent binary prediction (sigmoid, not softmax)
    because classes can spatially overlap (e.g. a road can simultaneously be
    drivable_area and contain lane dividers).

    Loss: element-wise sigmoid focal loss (γ=2, α=0.25), identical to
    BEVFusion's map head.  Plain BCE under-penalises easy negatives and
    fails on rare classes such as stop_line (~0.5% of BEV cells).

    Default map classes (nuScenes, 6 total):
        drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider

    Reference:
        BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye
        View Representation. Liu et al., ICRA 2023. arXiv:2205.13542.

    Args:
        in_channels (int): Input BEV feature channels (default 256).
        num_classes (int): Number of map classes (default 6).
        loss_seg (dict): Loss config.  Defaults to FocalLoss with sigmoid.
    """

    def __init__(
        self,
        in_channels=256,
        num_classes=6,
        loss_seg=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        init_cfg=None,
    ):
        super(MapSegHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.loss_seg = build_loss(loss_seg)

        # Three-layer conv decoder matching BEVFusion's lightweight head.
        # No upsampling: BEVFormer BEV features are already at 200×200.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, bev_feat):
        """
        Args:
            bev_feat (Tensor): (B, C, H, W) BEV feature map.
        Returns:
            Tensor: Raw logits (B, num_classes, H, W).
        """
        return self.conv_layers(bev_feat)

    @force_fp32(apply_to=('seg_logits',))
    def loss(self, seg_logits, gt_masks_bev):
        """Element-wise focal loss over all classes and spatial positions.

        Args:
            seg_logits (Tensor): Raw logits (B, num_classes, H, W).
            gt_masks_bev (Tensor): Binary GT masks  (B, num_classes, H, W).
        Returns:
            dict: {'loss_seg': scalar}
        """
        B, C, H, W = seg_logits.shape
        loss = self.loss_seg(
            seg_logits.reshape(B * C, H * W),
            gt_masks_bev.reshape(B * C, H * W).float(),
        )
        return dict(loss_seg=loss)

    def forward_train(self, bev_feat, gt_masks_bev):
        """Convenience wrapper: forward + loss in one call."""
        return self.loss(self.forward(bev_feat), gt_masks_bev)

    def get_seg_maps(self, bev_feat):
        """Inference: returns sigmoid-activated class probability maps."""
        return self.forward(bev_feat).sigmoid()
