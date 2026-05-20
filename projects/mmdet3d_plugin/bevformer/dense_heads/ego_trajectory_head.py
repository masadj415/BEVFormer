import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


class _EgoBEVCrossAttnLayer(nn.Module):
    """Ego query (1 token) cross-attending to flattened BEV features."""

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, bev):
        # query: [B, 1, C]  bev: [B, H*W, C]
        attn_out, _ = self.cross_attn(query, bev, bev)
        query = self.norm1(query + self.drop(attn_out))
        query = self.norm2(query + self.drop(self.ffn(query)))
        return query


@HEADS.register_module()
class EgoTrajectoryHead(nn.Module):
    """
    Ego trajectory prediction head.

    Predicts num_waypoints future ego positions (x, y) in the current ego
    frame, using:
      - A single learnable ego query
      - Canbus initialization (speed/yaw prior) to bootstrap the query
      - num_decoder_layers cross-attention layers over BEV features
      - A small trajectory MLP

    Temporal V-JEPA information flows in via bev_embed, which already
    encodes motion cues from the V-JEPA temporal slices (gated or mean fusion
    upstream).

    GT format: [B, num_waypoints, 2] with NaN for invalid future frames
    (e.g. end-of-scene).
    """

    def __init__(
        self,
        embed_dims=256,
        num_waypoints=6,
        canbus_dim=18,
        num_decoder_layers=2,
        num_heads=8,
        dropout=0.1,
        loss_weight=0.5,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.loss_weight = loss_weight

        # Canbus MLP: speed/yaw prior → query initialization offset
        self.canbus_mlp = nn.Sequential(
            nn.Linear(canbus_dim, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )

        # Learnable base ego query
        self.ego_query_embed = nn.Parameter(torch.zeros(1, embed_dims))
        nn.init.normal_(self.ego_query_embed, std=0.02)

        self.decoder_layers = nn.ModuleList(
            [_EgoBEVCrossAttnLayer(embed_dims, num_heads, dropout)
             for _ in range(num_decoder_layers)]
        )

        self.traj_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_waypoints * 2),
        )

    def forward(self, bev_embed, img_metas):
        """
        Args:
            bev_embed: [B, bev_h*bev_w, embed_dims]
            img_metas: list[dict] — each must contain 'can_bus' (18-D array)

        Returns:
            waypoints: [B, num_waypoints, 2] — predicted future (x, y)
        """
        B = len(img_metas)
        device = bev_embed.device

        # bev_embed from BEVFormerHead is [bev_h*bev_w, B, C] (seq-first).
        # Permute to [B, bev_h*bev_w, C] for batch-first attention.
        if bev_embed.shape[0] != B:
            bev_embed = bev_embed.permute(1, 0, 2).contiguous()

        can_bus = torch.stack(
            [torch.tensor(m['can_bus'], dtype=torch.float32) for m in img_metas]
        ).to(device)  # [B, 18]

        # [B, 1, C]: base ego query offset-initialized by canbus motion prior
        ego_q = (
            self.ego_query_embed.unsqueeze(0).expand(B, 1, -1)
            + self.canbus_mlp(can_bus).unsqueeze(1)
        )

        bev = bev_embed.float()
        for layer in self.decoder_layers:
            ego_q = layer(ego_q, bev)

        waypoints = self.traj_mlp(ego_q.squeeze(1))          # [B, 12]
        return waypoints.reshape(B, self.num_waypoints, 2)

    def loss(self, pred_waypoints, gt_future_ego):
        """
        Args:
            pred_waypoints: [B, num_waypoints, 2]
            gt_future_ego:  [B, num_waypoints, 2], NaN where invalid

        Returns:
            dict with 'loss_ego_traj'
        """
        gt = gt_future_ego.to(pred_waypoints.device, pred_waypoints.dtype)
        valid = torch.isfinite(gt).all(dim=-1)  # [B, num_waypoints]

        if not valid.any():
            return {'loss_ego_traj': pred_waypoints.sum() * 0.0}

        # ADE loss: mean Euclidean displacement per waypoint → units = meters.
        # Matches the ADE/FDE metrics reported at evaluation time.
        err = (pred_waypoints[valid] - gt[valid]).norm(dim=-1)  # [num_valid]
        loss = err.mean()
        return {'loss_ego_traj': self.loss_weight * loss}
