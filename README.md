# BEVFormer with Cached V-JEPA and DINO Features

This repository is a project fork of **BEVFormer** for camera-only 3D perception on nuScenes. The main idea is to replace the standard image backbone/FPN with **cached dense self-supervised visual tokens** and study whether frozen video/image representations can be adapted into a BEVFormer-style bird's-eye-view representation.

The project compares cached **V-JEPA** and **DINO** features, adds lightweight token-to-BEVFormer adapters, and extends the detector with optional multi-task heads for 3D detection, ego trajectory prediction, agent motion prediction, and HD-map segmentation.

<p align="center">
  <a href="assets/dino_scene481_thr035_10s.mp4">
    <img src="assets/dino_scene481_thr035_preview.gif" width="750">
  </a>
  <br>
  <em>DINO cached-feature predictions on nuScenes scene</em>
</p>

<p align="center">
  <a href="assets/vjepa-bev.mp4">
    <img src="assets/vjepa_scene481_thr035_preview.gif" width="750">
  </a>
  <br>
  <em>V-JEPA cached-feature predictions on nuScenes scene</em>
</p>

## Project overview

Standard BEVFormer uses RGB images as input:

```text
multi-camera images -> CNN/FPN backbone -> BEVFormer encoder/decoder -> 3D boxes
```

This project replaces the ResNet backbone with cached token features:

```text
cached V-JEPA / DINO HDF5 features
        -> camera-order alignment + geometry scaling
        -> temporal slice reduction / gated temporal fusion
        -> adapter
        -> BEVFormer spatial cross-attention + temporal BEV reasoning
        -> detection / trajectory / motion / map heads
```

The cached-feature setup makes it possible to test different frozen representation backbones without repeatedly recomputing dense features during BEVFormer training.


## Qualitative videos

The repository includes short qualitative visualizations under `assets/`. GitHub opens the MP4 files directly from the README links.

<p align="center">
  <a href="assets/dino_scene481_thr035_10s.mp4"><strong>Watch DINO qualitative video</strong></a>
  <br>
  <em>BEVFormer-DINO cached-feature predictions on a nuScenes validation scene.</em>
</p>

<p align="center">
  <a href="assets/vjepa_scene481_thr035_10s.mp4"><strong>Watch V-JEPA qualitative video</strong></a>
  <br>
  <em>BEVFormer-V-JEPA cached-feature predictions on a nuScenes validation scene.</em>
</p>

## Main contributions

## Repository structure

```text
.
├── projects/
│   ├── configs/bevformer/              # V-JEPA, DINO, baseline, and multi-head configs
│   └── mmdet3d_plugin/
│       ├── bevformer/detectors/        # BEVFormerVJepa and BEVFormerDino
│       ├── bevformer/dense_heads/      # detection, ego, motion, and map heads
│       └── datasets/pipelines/         # cached feature and map-mask loading
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── dist_train.sh
│   ├── dist_test.sh
│   └── analysis_tools/                 # visualization and analysis utilities
├── scripts/                            # cluster/HPC training and evaluation scripts; backbone feature caching script
├── assets/                             # qualitative figures and video demos
```

Important project files:

| File | Purpose |
|---|---|
| `projects/mmdet3d_plugin/bevformer/detectors/bevformer_vjepa.py` | BEVFormer variant that consumes cached V-JEPA features. |
| `projects/mmdet3d_plugin/bevformer/detectors/bevformer_dino.py` | BEVFormer variant that consumes cached DINO/generic dense tokens. |
| `projects/mmdet3d_plugin/datasets/pipelines/loading_vjepa.py` | HDF5 feature loader, camera alignment, and geometry scaling. |
| `projects/mmdet3d_plugin/datasets/pipelines/formatting_vjepa.py` | Formatter for cached feature tensors and extra targets. |
| `projects/mmdet3d_plugin/bevformer/dense_heads/ego_trajectory_head.py` | Future ego trajectory head. |
| `projects/mmdet3d_plugin/bevformer/dense_heads/motion_head.py` | Agent motion prediction head. |
| `projects/mmdet3d_plugin/bevformer/dense_heads/map_seg_head.py` | BEV HD-map segmentation head. |
| `projects/configs/bevformer/` | Main experiment configurations. |

## Cached feature format

The repository assumes that dense visual features have already been extracted and saved in HDF5 files. The expected layout is:

```text
<feature_file>.h5
└── vitb/
    ├── <sample_token_1> -> float16 array [6, num_tokens, 768]
    ├── <sample_token_2> -> float16 array [6, num_tokens, 768]
    └── ...
```

The first dimension corresponds to the six nuScenes cameras:

```python
CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
```

The loader reorders this fixed cached order to match the order in `img_filename`, so the feature tensor stays aligned with BEVFormer's camera matrices.

Typical token grids used in the experiments:

| Encoder | Image resolution | Token grid | Feature tensor per sample |
|---|---:|---:|---:|
| V-JEPA | 224 x 384 | 14 x 24 | `[6, T*14*24, 768]` |
| V-JEPA | 368 x 656 | 23 x 41 | `[6, T*23*41, 768]` |
| V-JEPA | 448 x 800 | 28 x 50 | `[6, T*28*50, 768]` |
| DINO | 368 x 640 | 23 x 40 | `[6, 23*40, 768]` |
| DINO | 448 x 800 | 28 x 50 | `[6, 28*50, 768]` |
| DINO | 896 x 1600 | 56 x 100 | `[6, 56*100, 768]` |

For V-JEPA, `T` is the number of temporal slices stored inside the cached feature tensor. The model supports `last`, `mean`, and `gated` temporal reduction. The gated variant requires at least two temporal slices.

## Installation

This codebase follows the original BEVFormer / MMDetection3D dependency stack. One working setup is:

```bash
conda create -n bevformer python=3.8 -y
conda activate bevformer

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# mmdetection3d version used by the original BEVFormer codebase
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
cd ..

pip install einops fvcore iopath==0.1.9 timm==0.6.13 numpy==1.19.5 \
  matplotlib==3.5.2 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
```

## Data preparation

Prepare the nuScenes dataset following the original BEVFormer instructions:

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus ./data
```

The expected directory structure is:

```text
data/
├── can_bus/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-trainval/
    ├── nuscenes_infos_temporal_train.pkl
    └── nuscenes_infos_temporal_val.pkl
```

For multi-task experiments with map and motion targets, the configs use augmented annotation files such as:

```text
nuscenes_infos_temporal_train_with_map_200_motion.pkl
nuscenes_infos_temporal_val_with_map_200_motion.pkl
```

Then, you can process the nuScenes dataset through the backbone and save the cached features:
1. Configure the resolution (must be divisible by patch size) and the batch size you want to process at in `scripts/cache_nuscenes_{vjepa,dino}.py`
2. Submit the caching job to SLURM cluster with `sbatch cache_nuscenes_{vjepa,dino}.sh`

This will generate the `XXX.h5` file to use for training.

## Training

A generic distributed training command is:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  tools/train.py \
  projects/configs/bevformer/<CONFIG>.py \
  --launcher pytorch \
  --work-dir work_dirs/<RUN_NAME>
```

Example: train the DINO cached-feature detection model at 448 x 800:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  tools/train.py \
  projects/configs/bevformer/bevformer_dino_448x800_bev200_q1_last_adapt4x512_8pts_bs4.py \
  --launcher pytorch \
  --work-dir work_dirs/bevformer_dino_448x800
```

For cluster runs, see the scripts in `scripts/`.

Evaluate a checkpoint on nuScenes validation:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  tools/test.py \
  projects/configs/bevformer/<CONFIG>.py \
  work_dirs/<RUN_NAME>/epoch_<N>.pth \
  --launcher pytorch \
  --eval bbox
```

The standard detection metrics are nuScenes NDS, mAP, mATE, mASE, mAOE, mAVE, and mAAE.

If ego or motion heads are enabled, the custom dataset evaluation also reports:

```text
ego/ADE, ego/FDE
motion/ADE, motion/FDE
```

## Main experiment configurations

| Config | Description |
|---|---|
| `bevformer_dino_448x800_bev200_q1_last_adapt4x512_8pts_bs4.py` | DINO cached features, 448 x 800, 28 x 50 tokens, detection. |
| `bevformer_dino_448x800_bev200_q4_allheads_groupdetr11_8pts_bs4.py` | DINO cached features with all heads and Group-DETR queries. |
| `bevformer_vjepa_448x800_bev200_q1_last_adapt4x512_8pts_bs4.py` | V-JEPA cached features, last temporal slice, 448 x 800. |
| `bevformer_vjepa_448x800_bev200_q1_last_adapt6x512_8pts_bs4.py` | V-JEPA cached features with a deeper adapter. |
| `bevformer_vjepa_224x384_bev200_q1_gated_adapter512_8pts_bs8.py` | V-JEPA cached features, 224 x 384, gated temporal fusion. |
| `bevformer_vjepa_448x800_bev200_q4_last_adapt4x512_motion_ego_8pts_bs4.py` | V-JEPA with detection, ego trajectory, and motion-related targets. |

## Validation results

The table below summarizes the best completed validation runs from the provided experiment summary. Metrics are nuScenes validation metrics.

| Run | Encoder | Resolution | Token grid | Temporal setup | NDS | mAP | mATE | mAOE |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `bevformer_dino_lets_try_448_800` | DINO | 448 x 800 | 28 x 50 | queue 4, last | **0.4664** | **0.3535** | 0.7215 | 0.4448 |
| `bevformer_dino_lets_try_896x1600` | DINO | 896 x 1600 | 56 x 100 | queue 4, last | 0.4563 | 0.3522 | 0.7596 | 0.4766 |
| `bevformer_vjepa_new_temporal_4_izar_good_one` | V-JEPA | 448 x 800 | 28 x 50 | queue 4, last | 0.4311 | 0.3141 | 0.7725 | 0.5541 |
| `bevformer_vjepa_gated_3gpu_a100_80_bs12_w0_fix1_new_cache_good_cameras_HIGH` | V-JEPA | 448 x 800 | 28 x 50 | gated | 0.3633 | 0.2527 | 0.8193 | 0.6313 |
| `bevformer_vjepa_gated_3gpu_a100_80_bs12_w0_fix1_new_cache_good_cameras` | V-JEPA | 368 x 656 | 23 x 41 | gated | 0.3475 | 0.2300 | 0.8714 | 0.6701 |
| `bevformer_vjepa_new_adapter_6blocks_3gpu_a100_80_bs12_w0_fix1_high` | V-JEPA | 448 x 800 | 28 x 50 | last | 0.3414 | 0.2456 | 0.8847 | 0.6524 |

The strongest result in these experiments is the DINO 448 x 800 model. The strongest V-JEPA detection result uses 448 x 800 cached features with a queue length of 4 and last-slice temporal reduction.


## Citation

If you use the BEVFormer backbone or codebase, please cite the original BEVFormer paper:

```bibtex
@article{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
```
