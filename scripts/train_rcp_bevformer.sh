#!/usr/bin/env bash
set -e

CONFIG=$1
GPUS=$2
WORK_DIR=$3

mkdir -p "$WORK_DIR"

LOGFILE="$WORK_DIR/runai_stdout_stderr_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOGFILE") 2>&1

cd /mnt/vilab/scratch/masha/flextok_RCP/BEVFormer

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export WANDB_DIR=/mnt/vilab/scratch/masha/wandb

echo "===== START ====="
date
hostname
pwd

echo "===== GPU INFO ====="
nvidia-smi

echo "===== PYTHON CUDA INFO ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "===== TRAIN COMMAND ====="
echo "CONFIG=$CONFIG"
echo "GPUS=$GPUS"
echo "WORK_DIR=$WORK_DIR"
echo "LOGFILE=$LOGFILE"

bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR"
