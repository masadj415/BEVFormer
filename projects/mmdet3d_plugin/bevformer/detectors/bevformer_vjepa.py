import torch
import torch.nn as nn

from mmcv.runner import auto_fp16
from mmcv.utils import build_from_cfg
from mmdet.models import DETECTORS, HEADS
from mmdet3d.core import bbox3d2result

from .bevformer import BEVFormer
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    
class ResidualConvBlock(nn.Module):
    """
    Residual spatial block on the V-JEPA token grid.

    Keeps the same shape:
        [B*num_cams, C, H, W] -> [B*num_cams, C, H, W]

    Important:
        GELU is used inside the block, but there is no activation after
        the residual addition. This keeps signed features for the transformer.
    """

    def __init__(self, channels=256, num_groups=32):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
        )

        # Start residual branch close to zero for stability.
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        return x + self.block(x)


class VJepaAdapter(nn.Module):
    """
    Residual adapter from cached V-JEPA dense tokens to BEVFormer image-feature format.

    Input:
        x: [B, num_cams, 768, H, W]

    Output:
        x: [B, num_cams, 256, H, W]

    Design:
        768 -> 512 -> 256 projection
        + several residual 3x3 conv blocks on the V-JEPA spatial grid
        + final 1x1 refinement without final activation

    Important:
        There is no ReLU/GELU at the final output, because BEVFormer attention
        should receive signed features, not only non-negative features.
    """

    def __init__(
        self,
        in_dim=768,
        out_dim=256,
        hidden_dim=512,
        num_blocks=4,
        num_groups=32,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.GELU(),

            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_dim),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualConvBlock(
                    channels=out_dim,
                    num_groups=num_groups,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, out_dim),
        )

    def forward(self, x):
        # x: [B, num_cams, C, H, W]
        B, N, C, H, W = x.shape

        x = x.reshape(B * N, C, H, W)

        # Cached V-JEPA features are float16, but adapter is safer in fp32.
        x = x.float()

        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.final(x)

        x = x.reshape(B, N, -1, H, W)
        return x

class TokenWiseTemporalGate(nn.Module):
    """
    Token-wise gated fusion between current and previous V-JEPA temporal slices.

    Input:
        curr: [B, num_cams, C, H, W]
        prev: [B, num_cams, C, H, W]

    Gate:
        G = sigmoid(f([F_curr, F_prev, F_curr - F_prev]))

    Output:
        fused = (1 - G) * F_curr + G * F_prev

    Here G is scalar per camera/spatial token:
        G: [B, num_cams, 1, H, W]

    It is broadcast over the feature channels.
    """

    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(3 * in_dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, curr, prev):
        # curr, prev: [B, num_cams, C, H, W]
        B, N, C, H, W = curr.shape

        x = torch.cat([curr, prev, curr - prev], dim=2)
        x = x.reshape(B * N, 3 * C, H, W).float()

        gate = self.gate(x)  # [B*N, 1, H, W]
        gate = gate.reshape(B, N, 1, H, W)

        fused = (1.0 - gate) * curr + gate * prev
        return fused

@DETECTORS.register_module()
class BEVFormerVJepa(BEVFormer):
    """
    BEVFormer variant that consumes cached V-JEPA features instead of images.

    Expected input img:
        [B, 6, 672, 768]

    where:
        6   = number of cameras
        672 = T * 14 * 24 V-JEPA tokens
        768 = ViT-B feature dimension

    This version supports:
        - last temporal slice
        - mean temporal fusion
        - token-wise gated temporal fusion
        - stronger convolutional adapter
    """

    def __init__(
        self,
        vjepa_in_dim=768,
        vjepa_out_dim=256,
        vjepa_h=14,
        vjepa_w=24,
        vjepa_temporal_reduce="last",
        vjepa_adapter_hidden_dim=512,
        vjepa_adapter_num_blocks=4,
        vjepa_gate_hidden_dim=256,
        ego_trajectory_head=None,
        motion_head=None,
        map_seg_head=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.vjepa_in_dim = vjepa_in_dim
        self.vjepa_out_dim = vjepa_out_dim
        self.vjepa_h = vjepa_h
        self.vjepa_w = vjepa_w
        self.vjepa_temporal_reduce = vjepa_temporal_reduce

        self.vjepa_adapter = VJepaAdapter(
            in_dim=vjepa_in_dim,
            out_dim=vjepa_out_dim,
            hidden_dim=vjepa_adapter_hidden_dim,
            num_blocks=vjepa_adapter_num_blocks,
        )
        if self.vjepa_temporal_reduce == "gated":
            self.vjepa_temporal_gate = TokenWiseTemporalGate(
                in_dim=vjepa_in_dim,
                hidden_dim=vjepa_gate_hidden_dim,
            )
        else:
            self.vjepa_temporal_gate = None

        self.ego_trajectory_head = (
            build_from_cfg(ego_trajectory_head, HEADS)
            if ego_trajectory_head is not None else None
        )
        self.motion_head = (
            build_from_cfg(motion_head, HEADS)
            if motion_head is not None else None
        )
        self.map_seg_head = (
            build_from_cfg(map_seg_head, HEADS)
            if map_seg_head is not None else None
        )

    def _prepare_vjepa_features(self, img):
        """
        Convert cached V-JEPA features to BEVFormer image feature map.

        Accepts:
            [B, 6, 672, 768]
            [B, len_queue, 6, 672, 768]

        Returns:
            [B, 6, 256, 14, 24]
        """

        if img is None:
            return None

        # If dataloader gives a queue dimension, take the current frame only.
        # The gated fusion below is over V-JEPA temporal slices inside the cache,
        # not over BEVFormer queue frames.
        if img.dim() == 5:
            img = img[:, -1]

        # If somehow a single unbatched sample appears.
        if img.dim() == 3:
            img = img.unsqueeze(0)

        if img.dim() != 4:
            raise ValueError(
                f"Expected V-JEPA features with shape [B, 6, tokens, 768], "
                f"but got {tuple(img.shape)}"
            )

        B, num_cams, num_tokens, C = img.shape

        if C != self.vjepa_in_dim:
            raise ValueError(
                f"Expected V-JEPA channel dim {self.vjepa_in_dim}, got {C}"
            )

        spatial_tokens = self.vjepa_h * self.vjepa_w

        if num_tokens % spatial_tokens != 0:
            raise ValueError(
                f"num_tokens={num_tokens} is not divisible by "
                f"vjepa_h*vjepa_w={spatial_tokens}"
            )

        T = num_tokens // spatial_tokens

        # [B, 6, tokens, 768]
        # -> [B, 6, T, 14, 24, 768]
        img = img.reshape(B, num_cams, T, self.vjepa_h, self.vjepa_w, C)

        if self.vjepa_temporal_reduce == "last":
            img = img[:, :, -1]  # [B, 6, 14, 24, 768]
            img = img.permute(0, 1, 4, 2, 3).contiguous()  # [B, 6, 768, 14, 24]

        elif self.vjepa_temporal_reduce == "mean":
            img = img.mean(dim=2)  # [B, 6, 14, 24, 768]
            img = img.permute(0, 1, 4, 2, 3).contiguous()  # [B, 6, 768, 14, 24]

        elif self.vjepa_temporal_reduce == "gated":
            if T < 2:
                raise ValueError(
                    "vjepa_temporal_reduce='gated' requires at least two V-JEPA "
                    f"temporal slices, but got T={T}"
                )

            prev = img[:, :, -2]  # [B, 6, 14, 24, 768]
            curr = img[:, :, -1]  # [B, 6, 14, 24, 768]

            prev = prev.permute(0, 1, 4, 2, 3).contiguous()  # [B, 6, 768, 14, 24]
            curr = curr.permute(0, 1, 4, 2, 3).contiguous()  # [B, 6, 768, 14, 24]

            img = self.vjepa_temporal_gate(curr, prev)       # [B, 6, 768, 14, 24]

        else:
            raise ValueError(
                f"Unknown vjepa_temporal_reduce={self.vjepa_temporal_reduce}"
            )

        # [B, 6, 768, 14, 24] -> [B, 6, 256, 14, 24]
        img = self.vjepa_adapter(img)

        return img

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """
        Instead of:
            images -> img_backbone -> img_neck

        we do:
            cached V-JEPA features -> reshape/fusion -> adapter

        Return format:
            list[Tensor]

        Since we only have one V-JEPA feature level, we return:
            [vjepa_feats]

        where:
            vjepa_feats: [B, num_cams, C, H, W]
        """

        vjepa_feats = self._prepare_vjepa_features(img)

        if vjepa_feats is None:
            return None

        return [vjepa_feats]

    @auto_fp16(apply_to=("img",))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(img, img_metas, len_queue=len_queue)

    @torch.no_grad()
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """
        Build previous BEV features from queued V-JEPA cached features.

        imgs_queue:
            [B, len_queue, 6, tokens, 768]

        img_metas_list:
            list over batch, each element is list of metadata over queue

        Returns:
            prev_bev from the last history frame
        """
        self.eval()

        prev_bev = None
        bs, len_queue = imgs_queue.shape[:2]

        for i in range(len_queue):
            img = imgs_queue[:, i, ...]  # [B, 6, tokens, 768]

            # Metadata for the i-th frame in the queue
            if isinstance(img_metas_list[0], list):
                img_metas = [each[i] for each in img_metas_list]
            elif isinstance(img_metas_list[0], dict) and i in img_metas_list[0]:
                img_metas = [each[i] for each in img_metas_list]
            else:
                raise ValueError(
                    "Expected img_metas to contain queue metadata, but got "
                    f"type={type(img_metas_list[0])}"
                )

            # Reset temporal memory at scene boundary
            if not img_metas[0].get("prev_bev_exists", True):
                prev_bev = None

            img_feats = self.extract_feat(img=img, img_metas=img_metas)

            prev_bev = self.pts_bbox_head(
                img_feats,
                img_metas,
                prev_bev=prev_bev,
                only_bev=True,
            )

        self.train()
        return prev_bev
    

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        gt_future_ego=None,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_masks_bev=None,
        prev_bev=None,
    ):
        """
        Detection + ego-trajectory + motion prediction training pass.

        Extra args vs. the base class:
            gt_future_ego:     [B, num_waypoints, 2] — future ego positions in
                               current ego frame (NaN for end-of-scene).
            gt_fut_traj:       list[Tensor[N, T, 2]] — per-agent future
                               trajectories in agent-local frame.
            gt_fut_traj_mask:  list[Tensor[N, T]]    — validity mask
                               (1 = valid step, 0 = agent exited scene).
        """
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        losses = self.pts_bbox_head.loss(
            gt_bboxes_3d, gt_labels_3d, outs, img_metas=img_metas
        )

        # Ego Trajectory Head ────────────────────────────────────
        if self.ego_trajectory_head is not None and gt_future_ego is not None:
            bev_embed = outs['bev_embed']          # [B, H*W, C]
            ego_wp = self.ego_trajectory_head(bev_embed, img_metas)
            losses.update(self.ego_trajectory_head.loss(ego_wp, gt_future_ego))

        # Agent Motion Head ───────────────────────────────────────
        if (
            self.motion_head is not None
            and gt_fut_traj is not None
            and gt_fut_traj_mask is not None
        ):
            query_feats = outs.get('query_feats')  # [B, 900, C]
            if query_feats is not None:
                motion_preds = self.motion_head(query_feats.float())
                last_cls  = outs['all_cls_scores'][-1].detach()
                last_bbox = outs['all_bbox_preds'][-1].detach()
                pos_inds, pos_gt_inds = self.pts_bbox_head.get_motion_matching(
                    last_cls, last_bbox, gt_bboxes_3d, gt_labels_3d
                )
                # Extract vx, vy from GT boxes if present; otherwise use zeros.
                # Some pipelines keep boxes as 7D and store velocity elsewhere,
                # so this avoids crashing when the tensor has no velocity columns.
                gt_velocities = []
                for boxes in gt_bboxes_3d:
                    box_tensor = boxes.tensor
                    if box_tensor.size(-1) >= 9:
                        gt_velocities.append(box_tensor[:, 7:9])
                    else:
                        gt_velocities.append(box_tensor.new_zeros((box_tensor.size(0), 2)))
                losses.update(
                    self.motion_head.loss(
                        motion_preds,
                        pos_inds,
                        pos_gt_inds,
                        gt_fut_traj,
                        gt_fut_traj_mask,
                        gt_velocities_list=gt_velocities,
                    )
                )

        # Map Segmentation Head ──────────────────────────────────
        if self.map_seg_head is not None and gt_masks_bev is not None:
            bev_embed = outs['bev_embed']  # [H*W, B, C] — sequence-first from transformer
            B = bev_embed.shape[1]
            bev_h = self.pts_bbox_head.bev_h
            bev_w = self.pts_bbox_head.bev_w
            bev_feat = bev_embed.permute(1, 2, 0).reshape(B, -1, bev_h, bev_w)
            losses.update(self.map_seg_head.forward_train(bev_feat, gt_masks_bev))

        return losses

    @auto_fp16(apply_to=("img", "points"))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        gt_future_ego=None,
        gt_fut_traj=None,
        gt_fut_traj_mask=None,
        gt_masks_bev=None,
    ):
        """
        V-JEPA temporal training path with optional auxiliary heads.

        If img has queue dimension:
            img: [B, queue_length, 6, tokens, 768]

        Then:
            previous frames -> obtain_history_bev -> prev_bev
            current frame + prev_bev -> detection / auxiliary losses
        """

        prev_bev = None

        # True BEVFormer temporal training path.
        if img is not None and img.dim() == 5 and img.size(1) > 1:
            len_queue = img.size(1)

            prev_img = img[:, :-1, ...]   # [B, queue-1, 6, tokens, 768]
            curr_img = img[:, -1, ...]    # [B, 6, tokens, 768]

            # Split metadata into previous-frame metadata and current-frame metadata.
            if isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], list):
                    prev_img_metas = [each[:-1] for each in img_metas]
                    curr_img_metas = [each[-1] for each in img_metas]

                elif isinstance(img_metas[0], dict) and 0 in img_metas[0]:
                    prev_img_metas = [
                        [each[i] for i in range(len_queue - 1)]
                        for each in img_metas
                    ]
                    curr_img_metas = [each[len_queue - 1] for each in img_metas]

                else:
                    raise ValueError(
                        "img has queue dimension, but img_metas does not look like queue metadata."
                    )
            else:
                raise ValueError("img has queue dimension, but img_metas is missing.")

            # Build history BEV from previous frames.
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

            # Reset prev_bev at scene boundary.
            if not curr_img_metas[0].get("prev_bev_exists", True):
                prev_bev = None

            img_metas = curr_img_metas
            img_feats = self.extract_feat(img=curr_img, img_metas=img_metas)

        else:
            # Single-frame fallback.
            if isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], list):
                    img_metas = [each[-1] for each in img_metas]
                elif isinstance(img_metas[0], dict) and 0 in img_metas[0]:
                    img_metas = [each[max(each.keys())] for each in img_metas]

            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            prev_bev = None

        losses = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_future_ego=gt_future_ego,
            gt_fut_traj=gt_fut_traj,
            gt_fut_traj_mask=gt_fut_traj_mask,
            gt_masks_bev=gt_masks_bev,
            prev_bev=prev_bev,
        )

        return losses

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """
        Detection + ego trajectory + agent motion inference.

        Stores per-sample keys in each result dict:
            ego_waypoints   [6, 2]            — predicted ego future (m, ego frame)
            motion_preds    [max_num, 6, 2]   — agent motion for top-K queries
            motion_pred_xy  [max_num, 2]      — decoded (x, y) of those queries (m, LiDAR frame)
            motion_scores   [max_num]         — max class score per query
        """
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        # Ego trajectory 
        if self.ego_trajectory_head is not None:
            bev_embed = outs['bev_embed']
            if bev_embed.shape[0] != len(img_metas):
                bev_embed = bev_embed.permute(1, 0, 2).contiguous()
            ego_wp = self.ego_trajectory_head(bev_embed.float(), img_metas)  # [B, 6, 2]
            for i, rd in enumerate(bbox_results):
                rd['ego_waypoints'] = ego_wp[i].cpu().numpy()

        # Agent motion 
        if self.motion_head is not None:
            query_feats = outs.get('query_feats')
            if query_feats is not None:
                motion_preds = self.motion_head(query_feats.float())  # [B, 900, 6, 2]
                cls_scores = outs['all_cls_scores'][-1]               # [B, 900, num_cls]
                bbox_preds  = outs['all_bbox_preds'][-1]              # [B, 900, 10]
                B           = cls_scores.shape[0]
                max_num     = self.pts_bbox_head.bbox_coder.max_num
                pc_range    = torch.tensor(
                    self.pts_bbox_head.bbox_coder.pc_range,
                    device=cls_scores.device, dtype=torch.float32,
                )

                # Top-K query selection — mirrors NMSFreeCoder.decode_single
                topk_scores, topk_flat = cls_scores.sigmoid().view(B, -1).topk(max_num)
                query_inds = topk_flat // cls_scores.shape[-1]  # [B, max_num]

                for i, rd in enumerate(bbox_results):
                    qi  = query_inds[i]                                     # [max_num]
                    xy  = denormalize_bbox(bbox_preds[i, qi], pc_range)[..., :2]
                    rd['motion_preds']   = motion_preds[i, qi].cpu().numpy()  # [max_num, 6, 2]
                    rd['motion_pred_xy'] = xy.cpu().numpy()                   # [max_num, 2]
                    rd['motion_scores']  = topk_scores[i].cpu().numpy()       # [max_num]

        return outs['bev_embed'], bbox_results

