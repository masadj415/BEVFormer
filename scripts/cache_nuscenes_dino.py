# cached_nuscenes_dino.py
import os
import sys
import time
import argparse
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
import torchvision.transforms as T
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/path/to/dinov3")
from dinov3.hub.backbones import dinov3_vitb16

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["TORCH_HOME"] = "/path/to/torch/cache"

NUSCENES_ROOT = f"/path/to/data/nuscenes"
DINOV3_WEIGHTS = f"/path/to/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

IMG_SIZE_H = 448
IMG_SIZE_W = 800
PATCH_SIZE = 16
PATCH_H = IMG_SIZE_H // PATCH_SIZE  # 56
PATCH_W = IMG_SIZE_W // PATCH_SIZE  # 100
EMBED_DIM = 768  # For Vit-B Change according to the specific DINOv3 variant/size used

OUTPUT_DIR = f"/path/to/dinov3_cache/vitb_{IMG_SIZE_H}x{IMG_SIZE_W}_parts"

BATCH_SIZE = 32  # Adjust to available memory
NUM_WORKERS = 8

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]

transform = T.Compose([
    T.Resize((IMG_SIZE_H, IMG_SIZE_W)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="train")

    args = parser.parse_args()

    if args.rank is None:
        args.rank = int(os.environ.get("RANK", 0))

    if args.world_size is None:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.local_rank is None:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    return args


def load_model(device):
    model = dinov3_vitb16(weights=DINOV3_WEIGHTS, pretrained=True)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


def get_frame(nusc, sample, cam):
    keyframe = nusc.get("sample_data", sample["data"][cam])
    path = Path(NUSCENES_ROOT) / keyframe["filename"]
    img = Image.open(path).convert("RGB")
    return transform(img)  # (3, H, W)


class NuScenesShardDataset(Dataset):
    def __init__(self, nusc, rank, world_size, existing_tokens=None):
        self.nusc = nusc

        if existing_tokens is None:
            existing_tokens = set()

        all_samples = []

        for idx, sample in enumerate(nusc.sample):
            if idx % world_size != rank:
                continue

            if sample["token"] in existing_tokens:
                continue

            all_samples.append(sample)

        self.samples = all_samples

        print(
            f"[rank {rank}] Samples left to cache: {len(self.samples)} "
            f"out of total {len(nusc.sample)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        token = sample["token"]

        frames = []
        for cam in CAMERAS:
            frame = get_frame(self.nusc, sample, cam)
            frames.append(frame)

        frames = torch.stack(frames)  # (6, 3, H, W)

        return token, frames


def cache_model(nusc, model, group, existing_tokens, rank, world_size, device):
    dataset = NuScenesShardDataset(
        nusc=nusc,
        rank=rank,
        world_size=world_size,
        existing_tokens=existing_tokens,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    t0 = time.time()

    for i, (tokens, frames) in enumerate(tqdm(loader, desc=f"rank {rank}")):
        B = frames.shape[0]

        # Flatten cameras into batch dimension: (B*6, 3, H, W)
        imgs = frames.view(B * 6, 3, IMG_SIZE_H, IMG_SIZE_W).to(device, non_blocking=True)

        with torch.inference_mode():
            with torch.amp.autocast("cuda"):
                feats = model.forward_features(imgs)
                out = feats["x_norm_patchtokens"]  # (B*6, PATCH_H*PATCH_W, EMBED_DIM)

            if i == 0:
                print(f"[rank {rank}] Raw DINOv3 patch token shape: {out.shape}")

            out = out.view(B, 6, *out.shape[1:])  # (B, 6, num_patches, EMBED_DIM)

        for j in range(B):
            token = tokens[j]

            if token in group:
                continue

            group.create_dataset(
                token,
                data=out[j].cpu().half().numpy(),
                compression="lzf",
            )

        if i % 100 == 0:
            elapsed = (time.time() - t0) / 3600
            done = min((i + 1) * BATCH_SIZE, len(dataset))
            total = len(dataset)

            print(
                f"[rank {rank}] [{done}/{total}] | "
                f"{elapsed:.2f}h elapsed"
            )


def main():
    args = parse_args()

    rank = args.rank
    world_size = args.world_size
    local_rank = args.local_rank
    split = args.split

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    nusc_version = "v1.0-trainval" if split == "train" else "v1.0-test"
    output_path = (
        f"{OUTPUT_DIR}/{split}_feat_vitb_{IMG_SIZE_H}x{IMG_SIZE_W}"
        f"_rank{rank}_of_{world_size}.h5"
    )

    print("=" * 80)
    print(f"Rank: {rank}")
    print(f"World size: {world_size}")
    print(f"Local rank: {local_rank}")
    print(f"Device: {device}")
    print(f"Split: {split} ({nusc_version})")
    print(f"Output: {output_path}")
    print("=" * 80)

    nusc = NuScenes(
        version=nusc_version,
        dataroot=NUSCENES_ROOT,
        verbose=True,
    )

    model = load_model(device)

    with h5py.File(output_path, "a") as f:
        grp = f.require_group("vitb")

        grp.attrs["IMG_SIZE_H"] = IMG_SIZE_H
        grp.attrs["IMG_SIZE_W"] = IMG_SIZE_W
        grp.attrs["PATCH_H"] = PATCH_H
        grp.attrs["PATCH_W"] = PATCH_W
        grp.attrs["EMBED_DIM"] = EMBED_DIM
        grp.attrs["RANK"] = rank
        grp.attrs["WORLD_SIZE"] = world_size
        grp.attrs["SPLIT"] = split

        existing_tokens = set(grp.keys())
        print(f"[rank {rank}] Already cached: {len(existing_tokens)} samples")

        cache_model(
            nusc=nusc,
            model=model,
            group=grp,
            existing_tokens=existing_tokens,
            rank=rank,
            world_size=world_size,
            device=device,
        )

    size_gb = Path(output_path).stat().st_size / 1e9
    print(f"[rank {rank}] Done! File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
