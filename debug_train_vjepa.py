import os
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
import projects.mmdet3d_plugin
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


CFG = "/mnt/vilab/scratch/masha/flextok_RCP/BEVFormer/projects/configs/bevformer/bevformer_adapter_jepa_rcp.py"
WORK_DIR = "/mnt/vilab/scratch/masha/bevformer_vjepa_debug"
NUM_ITERS = 10

os.makedirs(WORK_DIR, exist_ok=True)

cfg = Config.fromfile(CFG)
cfg.model.pretrained = None

print("Building dataset...")
dataset = build_dataset(cfg.data.train)
print("dataset length:", len(dataset))

print("Building model...")
model = build_model(
    cfg.model,
    train_cfg=cfg.get("train_cfg"),
    test_cfg=cfg.get("test_cfg"),
)

model.init_weights()
model.cuda()
model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
)

print("Starting mini training...")

for it in range(NUM_ITERS):
    data = collate([dataset[it]], samples_per_gpu=1)
    data = scatter(data, [0])[0]

    losses = model(return_loss=True, **data)
    loss = sum(v.mean() for v in losses.values())

    if not torch.isfinite(loss):
        print("Non-finite loss at iter", it, "loss:", loss)
        for k, v in losses.items():
            print(k, v)
        raise RuntimeError("Loss is NaN/inf")

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
    optimizer.step()

    print(
        f"iter {it+1:02d}/{NUM_ITERS} | "
        f"loss={loss.item():.4f} | "
        f"grad_norm={float(grad_norm):.4f}"
    )

ckpt_path = os.path.join(WORK_DIR, "debug_iter10.pth")
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "num_iters": NUM_ITERS,
        "cfg": CFG,
    },
    ckpt_path,
)

print("mini training finished ✅")
print("saved checkpoint:", ckpt_path)
