#!/bin/bash
#SBATCH --job-name=bev_ego_tracking
#SBATCH --time=12:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/tlphan/cs503/BEVFormer/slurm_logs/%x_%j.out
#SBATCH --error=/home/tlphan/cs503/BEVFormer/slurm_logs/%x_%j.err

WANDB_KEY=$1   # usage: sbatch run_vjepa_temporal.sh <your_wandb_api_key>

set -x
cat $0
export MASTER_PORT=25679
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=INFO

if [ -n "$WANDB_KEY" ]; then
  export WANDB_API_KEY=$WANDB_KEY
elif [ -f "$HOME/.wandb_apikey" ]; then
  export WANDB_API_KEY=$(cat $HOME/.wandb_apikey)
fi
export WANDB_MODE=${WANDB_API_KEY:+online}
export WANDB_MODE=${WANDB_MODE:-offline}
# Compute nodes can't spawn WandB's service subprocess.
# WANDB_START_METHOD=thread is no longer enough in wandb>=0.13 — the service
# layer is separate. WANDB_DISABLE_SERVICE=true bypasses it entirely.
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true

source /home/tlphan/miniconda3/etc/profile.d/conda.sh
conda activate /home/tlphan/miniconda3/envs/bev

cd /home/tlphan/cs503/BEVFormer
export PYTHONPATH=$PWD:$PWD/tools:${PYTHONPATH:-}

CONFIG=/home/tlphan/cs503/BEVFormer/projects/configs/bevformer/bevformer_vjepa_temporal_izar.py
WORKDIR=/scratch/izar/tlphan/work_dirs/vjepa_temporal_motion

mkdir -p /home/tlphan/cs503/BEVFormer/slurm_logs
mkdir -p $WORKDIR

RESUME_ARGS=""
if [ -f "$WORKDIR/latest.pth" ]; then
  RESUME_ARGS="--resume-from $WORKDIR/latest.pth"
else
  echo "WARNING: No checkpoint found in $WORKDIR — training from scratch"
fi

srun bash -c "
  TORCHRUN_ARGS=\"--node_rank=\${SLURM_PROCID} \
     --master_addr=\${MASTER_ADDR} \
     --master_port=\${MASTER_PORT} \
     --nnodes=\${SLURM_NNODES} \
     --nproc_per_node=2\"

  echo \${SLURM_PROCID}
  echo \${TORCHRUN_ARGS}
  echo \${SLURMD_NODENAME}

  torchrun \${TORCHRUN_ARGS} tools/train.py \
    $CONFIG \
    --launcher pytorch \
    --work-dir $WORKDIR \
    $RESUME_ARGS
" 2>&1 | tee $WORKDIR/train_${SLURM_JOB_ID}.log
