#!/bin/bash
#SBATCH --job-name=bev_ego_ft
#SBATCH --time=06:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/tlphan/cs503/BEVFormer/slurm_logs/%x_%j.out
#SBATCH --error=/home/tlphan/cs503/BEVFormer/slurm_logs/%x_%j.err

WANDB_KEY=$1

set -x
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

source /home/tlphan/miniconda3/etc/profile.d/conda.sh
conda activate /home/tlphan/miniconda3/envs/bev

cd /home/tlphan/cs503/BEVFormer
export PYTHONPATH=$PWD:$PWD/tools:$PYTHONPATH

CONFIG=projects/configs/bevformer/finetune_ego.py
WORKDIR=/scratch/izar/tlphan/work_dirs/bevformer_ego_finetune
PRETRAINED=/scratch/izar/tlphan/work_dirs/bevformer_small_map/latest.pth

if [ ! -f "$PRETRAINED" ]; then
  echo "ERROR: no checkpoint found at $PRETRAINED"
  exit 1
fi

mkdir -p /home/tlphan/cs503/BEVFormer/slurm_logs
mkdir -p $WORKDIR

RESUME_ARGS=""
if [ -f "$WORKDIR/latest.pth" ]; then
  RESUME_ARGS="--resume-from $WORKDIR/latest.pth"
else
  RESUME_ARGS="--load-from $PRETRAINED"
fi

srun bash -c "
  TORCHRUN_ARGS=\"--node_rank=\${SLURM_PROCID} \
     --master_addr=\${MASTER_ADDR} \
     --master_port=\${MASTER_PORT} \
     --nnodes=\${SLURM_NNODES} \
     --nproc_per_node=2\"

  torchrun \${TORCHRUN_ARGS} tools/train.py \
    $CONFIG \
    --launcher pytorch \
    --work-dir $WORKDIR \
    $RESUME_ARGS
" 2>&1 | tee $WORKDIR/train_${SLURM_JOB_ID}.log
