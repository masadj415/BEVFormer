import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS


@HEADS.register_module()
class EgoTrajectoryHead(BaseModule):
    """GRU-based ego trajectory waypoint predictor.

    Predicts num_waypoints future (x, y) positions in the current ego frame
    at 0.5s intervals: 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s.

    Architecture:
        BEV (B,C,H,W) → 2-layer conv → global avg pool → GRU × N → (B,N,2)

    nuScenes planning evaluation protocol (UniAD / VAD):
        L2 displacement error at t = 1s, 2s, 3s
        avg-L2 = mean of the three values

    Args:
        in_channels (int): BEV feature channels.
        hidden_dim (int): GRU hidden size.
        num_waypoints (int): Number of future waypoints (default 6 → 3s at 2Hz).
        loss_weight (float): Scalar multiplier on the regression loss.
    """

    def __init__(
        self,
        in_channels=256,
        hidden_dim=256,
        num_waypoints=6,
        loss_weight=1.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_waypoints = num_waypoints
        self.loss_weight = loss_weight

        # Encode BEV into a fixed scene embedding.
        # Two conv layers preserve local spatial context before pooling.
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                   # (B, hidden_dim)
        )

        # GRU cell: input = previous waypoint (x, y), state = scene embedding
        self.gru = nn.GRUCell(2, hidden_dim)
        self.wp_head = nn.Linear(hidden_dim, 2)

    def forward(self, bev_feat):
        """
        Args:
            bev_feat (Tensor): (B, C, H, W)
        Returns:
            Tensor: (B, num_waypoints, 2) — (x_forward, y_left) in ego frame
        """
        h = self.scene_encoder(bev_feat.float())  # (B, hidden_dim)

        waypoints = []
        wp = bev_feat.new_zeros(bev_feat.shape[0], 2)   # start at ego origin
        for _ in range(self.num_waypoints):
            h = self.gru(wp.float(), h)
            wp = self.wp_head(h)
            waypoints.append(wp)

        return torch.stack(waypoints, dim=1)  # (B, N, 2)

    @force_fp32(apply_to=('pred_waypoints',))
    def loss(self, pred_waypoints, gt_waypoints):
        """MSE regression on valid waypoints.

        GT missing future frames are padded with -999; those positions are
        excluded from the loss.

        Args:
            pred_waypoints (Tensor): (B, N, 2)
            gt_waypoints   (Tensor): (B, N, 2)
        Returns:
            dict: {'loss_ego_traj': scalar}
        """
        valid = gt_waypoints[..., 0] > -900   # (B, N) bool
        if not valid.any():
            return dict(loss_ego_traj=pred_waypoints.sum() * 0.0)

        loss = F.mse_loss(
            pred_waypoints[valid],
            gt_waypoints[valid].float(),
            reduction='mean',
        )
        return dict(loss_ego_traj=loss * self.loss_weight)

    def forward_train(self, bev_feat, gt_ego_waypoints):
        return self.loss(self.forward(bev_feat), gt_ego_waypoints)

    def get_waypoints(self, bev_feat):
        """Inference: returns waypoint tensor (B, N, 2)."""
        return self.forward(bev_feat)
