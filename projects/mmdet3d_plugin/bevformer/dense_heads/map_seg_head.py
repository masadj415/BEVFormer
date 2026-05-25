import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS


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
        # Store focal loss params directly — mmdet FocalLoss CUDA kernel
        # requires long class-index targets, incompatible with our binary
        # float masks. We implement sigmoid focal loss manually via F.bce.
        loss_cfg = loss_seg if isinstance(loss_seg, dict) else {}
        self.gamma = loss_cfg.get('gamma', 2.0)
        alpha = loss_cfg.get('alpha', 0.25)
        if isinstance(alpha, (list, tuple)):
            assert len(alpha) == num_classes, \
                f"alpha list length {len(alpha)} must match num_classes {num_classes}"
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            # scalar fallback — same behaviour as before
            self.register_buffer('alpha', torch.tensor([alpha] * num_classes, dtype=torch.float32))
        self.loss_weight = loss_cfg.get('loss_weight', 1.0)

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
        """Binary sigmoid focal loss over all classes and spatial positions.

        Fix 1: samples whose GT mask is entirely zero (no map annotations for
        that scene) are excluded from the loss.  Without this, the model learns
        that predicting all-zero is free, locking into a degenerate solution
        before ever seeing valid map data.  A graph-connected zero is returned
        when every sample in the batch is empty so that DDP can still aggregate
        gradients for all parameters.

        Fix 3: within the surviving valid samples, classes that are absent
        across the whole sub-batch (e.g. no stop-lines in this location) are
        also excluded.  Their absence should not push the head toward all-zero
        predictions on those thin classes.

        Args:
            seg_logits (Tensor): Raw logits (B, num_classes, H, W).
            gt_masks_bev (Tensor): Binary GT masks (B, num_classes, H, W).
        Returns:
            dict: {'loss_seg': scalar}
        """
        B, C, H, W = seg_logits.shape
        target = gt_masks_bev.float()  # (B, C, H, W)

        # Fix 1: per-sample validity — skip samples with all-zero GT maps.
        sample_valid = target.sum(dim=(1, 2, 3)) > 0  # (B,)
        if not sample_valid.any():
            # DDP requires all parameters to receive a gradient every step,
            # so return a zero that is still attached to the computation graph.
            return dict(loss_seg=seg_logits.sum() * 0.0)

        seg_logits = seg_logits[sample_valid]  # (V, C, H, W)
        target = target[sample_valid]          # (V, C, H, W)

        # Fix 3: per-class validity — skip classes absent in this sub-batch.
        class_valid = target.sum(dim=(0, 2, 3)) > 0  # (C,)

        alpha = self.alpha.view(1, C, 1, 1)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)

        pred_sigmoid = seg_logits.sigmoid()
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(seg_logits, target, reduction='none')

        # Average over present classes only so absent classes can't bias the
        # head toward all-zero predictions.
        loss = (focal_weight * bce)[:, class_valid].mean() * self.loss_weight
        return dict(loss_seg=loss)

    def forward_train(self, bev_feat, gt_masks_bev):
        """Convenience wrapper: forward + loss in one call."""
        return self.loss(self.forward(bev_feat), gt_masks_bev)

    def get_seg_maps(self, bev_feat):
        """Inference: returns sigmoid-activated class probability maps."""
        return self.forward(bev_feat).sigmoid()