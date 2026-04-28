import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
import projects.mmdet3d_plugin

cfg = Config.fromfile('projects/configs/bevformer/bevformer_tiny_vjepa_cached.py')

dataset = build_dataset(cfg.data.train)
loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    dist=False,
    shuffle=False,
)

batch = next(iter(loader))

model = build_model(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg')
)

model.init_weights()
model.cuda()
model.train()

img = batch["img"].data[0].cuda()

img_metas = batch["img_metas"].data[0]

gt_bboxes_3d = [
    b.to("cuda") if hasattr(b, "to") else b
    for b in batch["gt_bboxes_3d"].data[0]
]

gt_labels_3d = [
    x.cuda() if torch.is_tensor(x) else x
    for x in batch["gt_labels_3d"].data[0]
]

print("img:", img.shape, img.dtype)
print("num metas:", len(img_metas))
print("meta object:", img_metas[0])
print("gt labels:", gt_labels_3d[0].shape)

with torch.no_grad():
    losses = model(
        return_loss=True,
        img=img,
        img_metas=img_metas,
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d,
    )

print("LOSS KEYS:")
for k, v in losses.items():
    if torch.is_tensor(v):
        print(k, float(v.detach().cpu()))
    elif isinstance(v, list):
        vals = [float(x.detach().cpu()) for x in v if torch.is_tensor(x)]
        print(k, vals)
    else:
        print(k, type(v), v)

print("Forward/loss OK")
