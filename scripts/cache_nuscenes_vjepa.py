# cached_nuscenes_vjepa.py
import os
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

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["TORCH_HOME"] = "/path/to/torch/cache"

NUSCENES_ROOT = f"/path/to/data/nuscenes"

T_FRAMES = 2
IMG_SIZE_H = 448
IMG_SIZE_W = 800

OUTPUT_DIR = f"/path/to/vjepa_cache/vitb_{T_FRAMES}x{IMG_SIZE_H}x{IMG_SIZE_W}_parts"

BATCH_SIZE = 20  # Adjust to available memory
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
    encoder, _ = torch.hub.load(
        "facebookresearch/vjepa2",
        "vjepa2_1_vit_base_384",
        trust_repo=True,
        pretrained=True,
        force_reload=False,   # never re-download the hub repo
    )

    # IMPORTANT: no DataParallel here
    encoder = encoder.to(device).eval()

    for p in encoder.parameters():
        p.requires_grad_(False)

    return encoder


def get_clip(nusc, sample, cam, T=4):
    keyframe = nusc.get("sample_data", sample["data"][cam])
    frames = [keyframe]

    curr = keyframe
    for _ in range(T - 1): # Get exactly T-1 past frames
        if curr["prev"]:
            curr = nusc.get("sample_data", curr["prev"])
            frames.insert(0, curr)
        else:
            # If we run out of past frames (start of scene), duplicate the oldest one
            frames.insert(0, frames[0])

    imgs = []
    for f in frames:
        path = Path(NUSCENES_ROOT) / f["filename"]
        img = Image.open(path).convert("RGB")
        imgs.append(transform(img))

    clip = torch.stack(imgs)          # (T, 3, H, W)
    clip = clip.permute(1, 0, 2, 3)   # (3, T, H, W)

    return clip


class NuScenesShardDataset(Dataset):
    def __init__(self, nusc, rank, world_size, existing_tokens=None):
        self.nusc = nusc

        if existing_tokens is None:
            existing_tokens = set()

        all_samples = []

        for idx, sample in enumerate(nusc.sample):
            # This splits samples across GPUs
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

        clips = []
        for cam in CAMERAS:
            clip = get_clip(self.nusc, sample, cam, T_FRAMES)
            clips.append(clip)

        clips = torch.stack(clips)  # (6, 3, T, H, W)

        return token, clips


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

    for i, (tokens, clips) in enumerate(tqdm(loader, desc=f"rank {rank}")):
        B = clips.shape[0]

        clips = clips.view(
            B * 6, 3, T_FRAMES, IMG_SIZE_H, IMG_SIZE_W
        ).to(device, non_blocking=True)

        with torch.inference_mode():
            with torch.amp.autocast("cuda"):
                out = model(clips)

            if i == 0:
                print(f"[rank {rank}] Raw V-JEPA output shape: {out.shape}")

            out = out.view(B, 6, *out.shape[1:])

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
        f"{OUTPUT_DIR}/{split}_feat_vitb_{T_FRAMES}x{IMG_SIZE_H}x{IMG_SIZE_W}"
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

        grp.attrs["T_FRAMES"] = T_FRAMES
        grp.attrs["IMG_SIZE_H"] = IMG_SIZE_H
        grp.attrs["IMG_SIZE_W"] = IMG_SIZE_W
        grp.attrs["VJEPA_H"] = IMG_SIZE_H // 16
        grp.attrs["VJEPA_W"] = IMG_SIZE_W // 16
        grp.attrs["VJEPA_DIM"] = 768
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
