import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


@HEADS.register_module()
class MotionHead(nn.Module):
    """
    Per-agent motion prediction head.

    Piggybacks on the existing 900 DETR detection queries — no new queries.
    Applies a shared MLP to the last-decoder-layer query features to predict
    num_waypoints future positions in each agent's local coordinate frame.

    Agent-local frame convention:
        origin = detected agent's predicted center (x, y)
        x-axis = detected agent's heading (yaw direction)

    GT trajectories come from real future-frame annotations stored in the
    motion pkl (agent-local frame, origin = agent center, x = heading).
    Only moving agents (speed > min_agent_speed m/s) contribute to the loss.

    Supervision is applied only on Hungarian-matched queries (the pos_inds
    returned by the detection head's assigner).
    """

    def __init__(
        self,
        embed_dims=256,
        num_waypoints=6,
        future_dt=0.5,
        loss_weight=0.25,
        min_agent_speed=0.5,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.future_dt = future_dt
        self.loss_weight = loss_weight
        self.min_agent_speed = min_agent_speed

        self.motion_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_waypoints * 2),
        )

    def forward(self, query_feats):
        """
        Args:
            query_feats: [B, num_query, embed_dims]  (last decoder layer)

        Returns:
            motion: [B, num_query, num_waypoints, 2]  — local-frame offsets
        """
        B, N, C = query_feats.shape
        out = self.motion_mlp(query_feats)           # [B, N, num_waypoints*2]
        return out.reshape(B, N, self.num_waypoints, 2)

    def loss(
        self,
        motion_preds,
        pos_inds_list,
        pos_gt_inds_list,
        gt_fut_traj_list,
        gt_fut_traj_mask_list,
        gt_velocities_list=None,
    ):
        """
        Args:
            motion_preds:           [B, num_query, num_waypoints, 2]
            pos_inds_list:          list[Tensor] — matched query indices per batch
            pos_gt_inds_list:       list[Tensor] — corresponding GT box indices
            gt_fut_traj_list:       list[Tensor] — [N_gt, num_waypoints, 2] per batch,
                                    already in each agent's local frame
            gt_fut_traj_mask_list:  list[Tensor] — [N_gt, num_waypoints] per batch,
                                    1.0 = valid future step, 0.0 = agent exited scene
            gt_velocities_list:     list[Tensor] — [N_gt, 2] (vx, vy) in m/s global frame.
                                    When provided, only agents with speed > min_agent_speed
                                    contribute to the loss.

        Returns:
            dict with 'loss_motion'
        """
        device = motion_preds.device
        total_loss = motion_preds.new_zeros(1)
        num_pts = 0

        for b, (pos_inds, pos_gt_inds) in enumerate(
            zip(pos_inds_list, pos_gt_inds_list)
        ):
            if len(pos_inds) == 0:
                continue

            gt_traj = gt_fut_traj_list[b].to(device, motion_preds.dtype)   # [N, T, 2]
            gt_mask = gt_fut_traj_mask_list[b].to(device)                   # [N, T]

            matched_gt   = gt_traj[pos_gt_inds]   # [M, T, 2]
            matched_mask = gt_mask[pos_gt_inds]    # [M, T]   1=valid, 0=invalid

            pred = motion_preds[b, pos_inds]       # [M, T, 2]

            # Filter to moving agents only so stationary agents (the majority in
            # NuScenes) don't trivially drive the average loss to zero.
            if gt_velocities_list is not None:
                vels = gt_velocities_list[b].to(device, motion_preds.dtype)  # [N, 2]
                matched_vel = vels[pos_gt_inds]                               # [M, 2]
                speed = matched_vel.norm(dim=-1)                              # [M]
                moving = speed > self.min_agent_speed                         # [M] bool
                if not moving.any():
                    continue
                matched_gt   = matched_gt[moving]
                matched_mask = matched_mask[moving]
                pred         = pred[moving]

            # Broadcast mask over x/y dims and apply
            valid = matched_mask.unsqueeze(-1).expand_as(matched_gt).bool()
            if not valid.any():
                continue

            total_loss = total_loss + F.smooth_l1_loss(
                pred[valid], matched_gt[valid], reduction='sum'
            )
            num_pts += valid.sum().item()

        if num_pts == 0:
            return {'loss_motion': motion_preds.sum() * 0.0}

        return {'loss_motion': self.loss_weight * total_loss / num_pts}
