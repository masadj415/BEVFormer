# paste this in a quick test script
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
import torch

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# fake a GT box: car at (10, -5, 0), size (4, 2, 1.5), heading=0.3, vel=(5, 1)
box = torch.tensor([[10.0, -5.0, 0.0, 4.0, 2.0, 1.5, 0.3, 5.0, 1.0]])

normalized = normalize_bbox(box, pc_range)
print("normalized:", normalized)
# cx should be (10 - (-51.2)) / 102.4 = 0.599
# cy should be (-5 - (-51.2)) / 102.4 = 0.451
# vx should be 5/10 = 0.5

recovered = denormalize_bbox(normalized, pc_range)
print("recovered:", recovered)
# should match original box values closely
print("max error:", (box[..., :3] - recovered[..., :3]).abs().max())