import torch
import torch.nn as nn

from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS

from .bevformer import BEVFormer
class VJepaAdapter(nn.Module):
    """
    Stronger adapter from cached V-JEPA dense tokens to BEVFormer image-feature format.

    Input:
        x: [B, num_cams, 768, 14, 24]

    Output:
        x: [B, num_cams, 256, 14, 24]

    Difference from the old adapter:
        old: 768 -> 256 using 1x1 conv
        new: 768 -> 512 -> 512 -> 256 using 3x3 convs

    The 3x3 convolutions allow neighboring V-JEPA tokens on the 14x24 grid
    to exchange local spatial information before BEVFormer cross-attention.
    """

    def __init__(self, in_dim=768, out_dim=256, hidden_dim=512, num_groups=32):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, num_cams, C, H, W]
        B, N, C, H, W = x.shape

        x = x.reshape(B * N, C, H, W)

        # Cached V-JEPA features are float16, but adapter is safer in fp32.
        x = x.float()
        x = self.proj(x)

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
    ):
        """
        V-JEPA training path.

        This still does not use BEVFormer prev_bev temporal memory.
        Temporal fusion happens inside cached V-JEPA tokens when
        vjepa_temporal_reduce='gated'.
        """

        # If img_metas comes as queue metadata, keep only the current frame metadata.
        if isinstance(img_metas, list) and len(img_metas) > 0:
            if isinstance(img_metas[0], list):
                img_metas = [each[-1] for each in img_metas]
            elif isinstance(img_metas[0], dict) and 0 in img_metas[0]:
                img_metas = [each[max(each.keys())] for each in img_metas]

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()

        losses_pts = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            prev_bev=None,
        )

        losses.update(losses_pts)
        return losses
    
