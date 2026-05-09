import torch 


def normalize_bbox(bboxes, pc_range):
    cx = (bboxes[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])  # → [0,1]
    cy = (bboxes[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])  # → [0,1]
    cz = (bboxes[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])  # → [0,1]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]

    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] / 10.0   # normalize by max velocity
        vy = bboxes[..., 8:9] / 10.0
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # undo [0,1] normalization back to metric coordinates
    cx = normalized_bboxes[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    cy = normalized_bboxes[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    cz = normalized_bboxes[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2]

    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9] * 10.0   # undo velocity normalization
        vy = normalized_bboxes[..., 9:10] * 10.0
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes