#!/bin/bash
set -euo pipefail

cd /mnt/vilab/scratch/masha/flextok_RCP/BEVFormer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export USER=mduric
export LOGNAME=mduric
export HOME=/mnt/vilab/scratch/masha

export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR=/mnt/vilab/scratch/masha/.cache/matplotlib

# Helps with CUDA memory fragmentation.
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

CONFIG=/mnt/vilab/scratch/masha/flextok_RCP/BEVFormer/projects/configs/bevformer/bevformer_dino_all_heads.py
WORKDIR=/mnt/vilab/scratch/masha/work_dirs/bevformer_dino_all_heads

# Resume from the last checkpoint in THIS workdir.
# If latest.pth points to epoch_2.pth, this resumes from epoch 2 and continues at epoch 3.
RESUME=$WORKDIR/latest.pth

mkdir -p "$WORKDIR"

echo "===== CHECKS ====="
echo "HOME=$HOME"
echo "USER=$USER"
echo "LOGNAME=$LOGNAME"
echo "CONFIG=$CONFIG"
echo "WORKDIR=$WORKDIR"
echo "RESUME=$RESUME"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config does not exist: $CONFIG"
  exit 1
fi

echo "========== SYSTEM DEBUG =========="
echo "Date:"
date

echo "Hostname:"
hostname

echo "Visible CPU cores:"
nproc

echo "Memory:"
free -h

echo "/dev/shm:"
df -h /dev/shm

echo "Cgroup CPU limits:"
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true

echo "HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-<unset>}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"

echo "========== CPU DEBUG =========="
echo "nproc:"
nproc

echo "lscpu summary:"
lscpu | grep -E 'CPU\(s\)|Thread|Core|Socket' || true

echo "cpuset effective:"
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true

echo "cpu.max:"
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true

echo "computed CPU quota:"


echo "========== START TRAINING =========="

if [ -f "$HOME/.netrc" ]; then
  echo ".netrc exists"
else
  echo "ERROR: .netrc does not exist at $HOME/.netrc"
  exit 1
fi

echo "===== GPU INFO ====="
nvidia-smi

LOGFILE="$WORKDIR/train_resume_no_eval_$(date +%Y%m%d_%H%M%S).log"

set +e

python -m torch.distributed.launch \
  --nproc_per_node=3 \
  --master_port=29503 \
  tools/train.py \
  "$CONFIG" \
  --launcher pytorch \
  --no-validate \
  --cfg-options \
    data.samples_per_gpu=4 \
    data.workers_per_gpu=6 \
  2>&1 | tee "$LOGFILE"

STATUS=${PIPESTATUS[0]}

set -e
exit "$STATUS"