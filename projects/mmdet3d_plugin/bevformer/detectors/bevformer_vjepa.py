import torch
import torch.nn as nn

from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS

from .bevformer import BEVFormer


class ResidualConvBlock(nn.Module):
    """
    Residual spatial block on the V-JEPA token grid.

    Keeps shape:
        [B*num_cams, C, H, W] -> [B*num_cams, C, H, W]

    There is no activation after the residual addition, so the output remains
    signed and suitable for transformer attention.
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

        # Start residual branch close to zero for stable training.
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
        + num_blocks residual 3x3 conv blocks
        + final 1x1 refinement without final activation
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

        # Cached V-JEPA features are usually float16, but adapter is safer in fp32.
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
    BEVFormer variant that consumes cached V-JEPA features instead of RGB images.

    Supports:
        - temporal reduce: last / mean / gated
        - residual V-JEPA adapter
        - detection loss
        - optional map segmentation loss through gt_masks_bev
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

    def _prepare_vjepa_features(self, img):
        """
        Convert cached V-JEPA features to BEVFormer image feature map.

        Accepts:
            [B, 6, tokens, 768]
            [B, len_queue, 6, tokens, 768]

        Returns:
            [B, 6, 256, vjepa_h, vjepa_w]
        """

        if img is None:
            return None

        # If dataloader gives queue dimension, take current frame only.
        if img.dim() == 5:
            img = img[:, -1]

        # Single unbatched sample fallback.
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
        # -> [B, 6, T, H, W, 768]
        img = img.reshape(B, num_cams, T, self.vjepa_h, self.vjepa_w, C)

        if self.vjepa_temporal_reduce == "last":
            img = img[:, :, -1]  # [B, 6, H, W, 768]
            img = img.permute(0, 1, 4, 2, 3).contiguous()

        elif self.vjepa_temporal_reduce == "mean":
            img = img.mean(dim=2)  # [B, 6, H, W, 768]
            img = img.permute(0, 1, 4, 2, 3).contiguous()

        elif self.vjepa_temporal_reduce == "gated":
            if T < 2:
                raise ValueError(
                    "vjepa_temporal_reduce='gated' requires at least two V-JEPA "
                    f"temporal slices, but got T={T}"
                )

            prev = img[:, :, -2]  # [B, 6, H, W, 768]
            curr = img[:, :, -1]  # [B, 6, H, W, 768]

            prev = prev.permute(0, 1, 4, 2, 3).contiguous()
            curr = curr.permute(0, 1, 4, 2, 3).contiguous()

            img = self.vjepa_temporal_gate(curr, prev)

        else:
            raise ValueError(
                f"Unknown vjepa_temporal_reduce={self.vjepa_temporal_reduce}"
            )

        # [B, 6, 768, H, W] -> [B, 6, 256, H, W]
        img = self.vjepa_adapter(img)

        return img

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """
        Replace RGB backbone/FPN with:
            cached V-JEPA features -> temporal reduce -> adapter

        Return format:
            list[Tensor], one tensor per feature level.

        Since V-JEPA gives one level:
            [vjepa_feats]
        """

        vjepa_feats = self._prepare_vjepa_features(img)

        if vjepa_feats is None:
            return None

        return [vjepa_feats]

    @auto_fp16(apply_to=("img",))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(img, img_metas, len_queue=len_queue)

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
        gt_masks_bev=None,
        **kwargs,
    ):
        """
        V-JEPA training path with:
            - 3D detection loss
            - optional map segmentation loss if gt_masks_bev is provided
        """

        # If img_metas comes as queue metadata, keep only current frame metadata.
        if isinstance(img_metas, list) and len(img_metas) > 0:
            if isinstance(img_metas[0], list):
                img_metas = [each[-1] for each in img_metas]
            elif isinstance(img_metas[0], dict) and 0 in img_metas[0]:
                img_metas = [each[max(each.keys())] for each in img_metas]

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()

        # Detection forward.
        outs = self.pts_bbox_head(
            img_feats,
            img_metas,
            prev_bev=None,
        )

        losses_pts = self.pts_bbox_head.loss(
            gt_bboxes_3d,
            gt_labels_3d,
            outs,
            img_metas=img_metas,
        )
        losses.update(losses_pts)

        # Optional map segmentation loss.
        if (
            hasattr(self, "map_seg_head")
            and self.map_seg_head is not None
            and gt_masks_bev is not None
        ):
            if "bev_embed" not in outs:
                raise KeyError(
                    "Expected outs['bev_embed'] for map segmentation, "
                    f"but got keys: {list(outs.keys())}"
                )

            bev_embed = outs["bev_embed"]  # [bev_h*bev_w, B, C]

            B = bev_embed.shape[1]
            bev_h = self.pts_bbox_head.bev_h
            bev_w = self.pts_bbox_head.bev_w

            bev_feat = bev_embed.permute(1, 2, 0).reshape(
                B, -1, bev_h, bev_w
            )

            losses_seg = self.map_seg_head.forward_train(
                bev_feat,
                gt_masks_bev,
            )
            losses.update(losses_seg)

        return losses