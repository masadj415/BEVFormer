import torch
import torch.nn as nn

from mmcv.runner import auto_fp16
from mmcv.utils import build_from_cfg
from mmdet.models import DETECTORS, HEADS
from mmdet3d.core import bbox3d2result

from .bevformer import BEVFormer
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


class ResidualConvBlock(nn.Module):
    def __init__(self, channels=256, num_groups=32):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
        )

        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        return x + self.block(x)


class DinoAdapter(nn.Module):
    """
    Adapter from cached DINO / generic dense tokens to BEVFormer image-feature format.

    Input:
        x: [B, num_cams, C_in, H, W]

    Output:
        x: [B, num_cams, C_out, H, W]
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
        x = x.float()

        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.final(x)

        x = x.reshape(B, N, -1, H, W)
        return x


class TokenWiseTemporalGate(nn.Module):
    """
    Optional gated fusion if input features ever contain T>1 temporal slices.
    For DINO-only with [6, 1400, 768], T=1, so this is not used.
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

        gate = self.gate(x)
        gate = gate.reshape(B, N, 1, H, W)

        fused = (1.0 - gate) * curr + gate * prev
        return fused


@DETECTORS.register_module()
class BEVFormerDino(BEVFormer):
    """
    BEVFormer variant that consumes cached DINO/generic features instead of images.

    Expected DINO-only input per frame:
        [B, 6, 1400, 768]

    where:
        6    = cameras
        1400 = 28 * 50 spatial tokens for 448x800
        768  = feature dimension

    It also supports queue input:
        [B, queue_length, 6, 1400, 768]

    For queue_length > 1:
        previous frames -> obtain_history_bev -> prev_bev
        current frame + prev_bev -> detection / auxiliary losses
    """

    def __init__(
        self,
        # Keep vjepa_* names for config compatibility.
        vjepa_in_dim=768,
        vjepa_out_dim=256,
        vjepa_h=28,
        vjepa_w=50,
        vjepa_temporal_reduce="last",
        vjepa_adapter_hidden_dim=512,
        vjepa_adapter_num_blocks=4,
        vjepa_gate_hidden_dim=256,
        ego_trajectory_head=None,
        motion_head=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.feature_in_dim = vjepa_in_dim
        self.feature_out_dim = vjepa_out_dim
        self.feature_h = vjepa_h
        self.feature_w = vjepa_w
        self.temporal_reduce = vjepa_temporal_reduce

        self.feature_adapter = DinoAdapter(
            in_dim=vjepa_in_dim,
            out_dim=vjepa_out_dim,
            hidden_dim=vjepa_adapter_hidden_dim,
            num_blocks=vjepa_adapter_num_blocks,
        )

        if self.temporal_reduce == "gated":
            self.temporal_gate = TokenWiseTemporalGate(
                in_dim=vjepa_in_dim,
                hidden_dim=vjepa_gate_hidden_dim,
            )
        else:
            self.temporal_gate = None

        self.ego_trajectory_head = (
            build_from_cfg(ego_trajectory_head, HEADS)
            if ego_trajectory_head is not None else None
        )
        self.motion_head = (
            build_from_cfg(motion_head, HEADS)
            if motion_head is not None else None
        )

    def _prepare_cached_features(self, img):
        """
        Convert cached DINO / generic dense tokens to BEVFormer image feature map.

        Accepts:
            [B, 6, H*W, C]
            [B, len_queue, 6, H*W, C] as fallback

        Returns:
            [B, 6, 256, H, W]
        """

        if img is None:
            return None

        # Fallback only. In real temporal training, forward_train already splits queue.
        if img.dim() == 5:
            img = img[:, -1]

        if img.dim() == 3:
            img = img.unsqueeze(0)

        if img.dim() != 4:
            raise ValueError(
                f"Expected cached features with shape [B, 6, tokens, C], "
                f"but got {tuple(img.shape)}"
            )

        B, num_cams, num_tokens, C = img.shape

        if C != self.feature_in_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_in_dim}, got {C}"
            )

        spatial_tokens = self.feature_h * self.feature_w

        if num_tokens % spatial_tokens != 0:
            raise ValueError(
                f"num_tokens={num_tokens} is not divisible by "
                f"feature_h*feature_w={spatial_tokens}"
            )

        T = num_tokens // spatial_tokens

        # [B, 6, tokens, C] -> [B, 6, T, H, W, C]
        img = img.reshape(B, num_cams, T, self.feature_h, self.feature_w, C)

        if T == 1:
            # DINO-only case: [6, 1400, 768] => T=1.
            img = img[:, :, 0]
            img = img.permute(0, 1, 4, 2, 3).contiguous()

        elif self.temporal_reduce == "last":
            img = img[:, :, -1]
            img = img.permute(0, 1, 4, 2, 3).contiguous()

        elif self.temporal_reduce == "mean":
            img = img.mean(dim=2)
            img = img.permute(0, 1, 4, 2, 3).contiguous()

        elif self.temporal_reduce == "gated":
            if T < 2:
                raise ValueError(
                    "temporal_reduce='gated' requires at least two temporal slices, "
                    f"but got T={T}."
                )

            prev = img[:, :, -2]
            curr = img[:, :, -1]

            prev = prev.permute(0, 1, 4, 2, 3).contiguous()
            curr = curr.permute(0, 1, 4, 2, 3).contiguous()

            img = self.temporal_gate(curr, prev)

        else:
            raise ValueError(f"Unknown temporal_reduce={self.temporal_reduce}")

        img = self.feature_adapter(img)
        return img

    def extract_img_feat(self, img, img_metas, len_queue=None):
        feats = self._prepare_cached_features(img)

        if feats is None:
            return None

        return [feats]

    @auto_fp16(apply_to=("img",))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(img, img_metas, len_queue=len_queue)

    @torch.no_grad()
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """
        Build previous BEV features from queued cached features.

        imgs_queue:
            [B, len_queue, 6, tokens, C]

        img_metas_list:
            list over batch, each element is list of metadata over queue
        """
        was_training = self.training
        self.eval()

        prev_bev = None
        _, len_queue = imgs_queue.shape[:2]

        for i in range(len_queue):
            img = imgs_queue[:, i, ...]

            if isinstance(img_metas_list[0], list):
                img_metas = [each[i] for each in img_metas_list]
            elif isinstance(img_metas_list[0], dict) and i in img_metas_list[0]:
                img_metas = [each[i] for each in img_metas_list]
            else:
                raise ValueError(
                    "Expected queued img_metas, but got "
                    f"type={type(img_metas_list[0])}"
                )

            if not img_metas[0].get("prev_bev_exists", True):
                prev_bev = None

            img_feats = self.extract_feat(img=img, img_metas=img_metas)

            prev_bev = self.pts_bbox_head(
                img_feats,
                img_metas,
                prev_bev=prev_bev,
                only_bev=True,
            )

        if was_training:
            self.train()
        else:
            self.eval()

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
        prev_bev=None,
    ):
        outs = self.pts_bbox_head(
            pts_feats,
            img_metas,
            prev_bev=prev_bev,
        )

        losses = self.pts_bbox_head.loss(
            gt_bboxes_3d,
            gt_labels_3d,
            outs,
            img_metas=img_metas,
        )

        if self.ego_trajectory_head is not None and gt_future_ego is not None:
            bev_embed = outs["bev_embed"]
            ego_wp = self.ego_trajectory_head(bev_embed, img_metas)
            losses.update(self.ego_trajectory_head.loss(ego_wp, gt_future_ego))

        if (
            self.motion_head is not None
            and gt_fut_traj is not None
            and gt_fut_traj_mask is not None
        ):
            query_feats = outs.get("query_feats")

            if query_feats is not None:
                motion_preds = self.motion_head(query_feats.float())

                last_cls = outs["all_cls_scores"][-1].detach()
                last_bbox = outs["all_bbox_preds"][-1].detach()

                pos_inds, pos_gt_inds = self.pts_bbox_head.get_motion_matching(
                    last_cls,
                    last_bbox,
                    gt_bboxes_3d,
                    gt_labels_3d,
                )

                gt_velocities = []
                for boxes in gt_bboxes_3d:
                    box_tensor = boxes.tensor
                    if box_tensor.size(-1) >= 9:
                        gt_velocities.append(box_tensor[:, 7:9])
                    else:
                        gt_velocities.append(
                            box_tensor.new_zeros((box_tensor.size(0), 2))
                        )

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
    ):
        prev_bev = None

        if img is not None and img.dim() == 5 and img.size(1) > 1:
            len_queue = img.size(1)

            prev_img = img[:, :-1, ...]
            curr_img = img[:, -1, ...]

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
                        "img has queue dimension, but img_metas does not look "
                        "like queue metadata."
                    )
            else:
                raise ValueError("img has queue dimension, but img_metas is missing.")

            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

            if not curr_img_metas[0].get("prev_bev_exists", True):
                prev_bev = None

            img_metas = curr_img_metas
            img_feats = self.extract_feat(img=curr_img, img_metas=img_metas)

        else:
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
            prev_bev=prev_bev,
        )

        return losses

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs,
            img_metas,
            rescale=rescale,
        )

        bbox_results = [
            bboax3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        if self.ego_trajectory_head is not None:
            bev_embed = outs["bev_embed"]
            if bev_embed.shape[0] != len(img_metas):
                bev_embed = bev_embed.permute(1, 0, 2).contiguous()

            ego_wp = self.ego_trajectory_head(bev_embed.float(), img_metas)

            for i, rd in enumerate(bbox_results):
                rd["ego_waypoints"] = ego_wp[i].cpu().numpy()

        if self.motion_head is not None:
            query_feats = outs.get("query_feats")

            if query_feats is not None:
                motion_preds = self.motion_head(query_feats.float())
                cls_scores = outs["all_cls_scores"][-1]
                bbox_preds = outs["all_bbox_preds"][-1]

                B = cls_scores.shape[0]
                max_num = self.pts_bbox_head.bbox_coder.max_num

                pc_range = torch.tensor(
                    self.pts_bbox_head.bbox_coder.pc_range,
                    device=cls_scores.device,
                    dtype=torch.float32,
                )

                topk_scores, topk_flat = cls_scores.sigmoid().view(B, -1).topk(max_num)
                query_inds = topk_flat // cls_scores.shape[-1]

                for i, rd in enumerate(bbox_results):
                    qi = query_inds[i]
                    xy = denormalize_bbox(bbox_preds[i, qi], pc_range)[..., :2]

                    rd["motion_preds"] = motion_preds[i, qi].cpu().numpy()
                    rd["motion_pred_xy"] = xy.cpu().numpy()
                    rd["motion_scores"] = topk_scores[i].cpu().numpy()

        return outs["bev_embed"], bbox_results