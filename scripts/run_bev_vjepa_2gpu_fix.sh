#!/bin/bash
set -euo pipefail

cd /mnt/vilab/scratch/masha/flextok_RCP/BEVFormer

export USER=mduric
export LOGNAME=mduric
export HOME=/mnt/vilab/scratch/masha

export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR=/mnt/vilab/scratch/masha/.cache/matplotlib

# Optional, helps with CUDA memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

mkdir -p "$MPLCONFIGDIR"

export PATH=/mnt/vilab/scratch/masha/.local/bin:${PATH:-}
export PYTHONPATH=/mnt/vilab/scratch/masha/.local/lib/python3.7/site-packages:/mnt/vilab/scratch/masha/flextok_RCP/BEVFormer:/workspace/mmdetection3d:${PYTHONPATH:-}

unset WANDB_API_KEY

export WANDB_DIR=/mnt/vilab/scratch/masha/wandb
export WANDB_CACHE_DIR=/mnt/vilab/scratch/masha/.cache/wandb
export WANDB_CONFIG_DIR=/mnt/vilab/scratch/masha/.config/wandb
export WANDB_DATA_DIR=/mnt/vilab/scratch/masha/.local/share/wandb
export WANDB_ENTITY=masa-duric-epfl
export WANDB_MODE=online
export WANDB_START_METHOD=thread

git config --global --add safe.directory /mnt/vilab/scratch/masha/flextok_RCP/BEVFormer || true

mkdir -p "$WANDB_DIR"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$WANDB_CONFIG_DIR"
mkdir -p "$WANDB_DATA_DIR"

chmod 600 /mnt/vilab/scratch/masha/.netrc || true

CONFIG=/mnt/vilab/scratch/masha/flextok_RCP/BEVFormer/projects/configs/bevformer/bevformer_adapter_jepa_rcp.py
WORKDIR=/mnt/vilab/scratch/masha/work_dirs/bevformer_vjepa_gated_3gpu_a100_80_bs12_w0_fix1

mkdir -p "$WORKDIR"

echo "===== CHECKS ====="
echo "HOME=$HOME"
echo "USER=$USER"
echo "LOGNAME=$LOGNAME"
echo "CONFIG=$CONFIG"
echo "WORKDIR=$WORKDIR"

if [ -f "$HOME/.netrc" ]; then
  echo ".netrc exists"
else
  echo "ERROR: .netrc does not exist at $HOME/.netrc"
  exit 1
fi

echo "===== GPU INFO ====="
nvidia-smi

LOGFILE="$WORKDIR/train_$(date +%Y%m%d_%H%M%S).log"

set +e

timeout --signal=TERM --kill-after=5m 10h \
python -m torch.distributed.launch \
  --nproc_per_node=3 \
  --master_port=29503 \
  tools/train.py \
  "$CONFIG" \
  --launcher pytorch \
  --work-dir "$WORKDIR" \
  --cfg-options \
    data.samples_per_gpu=12 \
    data.workers_per_gpu=0 \
  2>&1 | tee "$LOGFILE"

STATUS=${PIPESTATUS[0]}

set -e

if [ "$STATUS" -eq 124 ]; then
  echo "===== TIME LIMIT REACHED: 10 hours. Stopping job cleanly. =====" | tee -a "$LOGFILE"
  exit 0
else
  exit "$STATUS"
fi