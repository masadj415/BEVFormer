#!/bin/bash
#SBATCH --job-name=bevformer_baseline
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/izar/mduric/slurm_logs/%x_%j.out

source /home/mduric/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/izar/mduric/conda_envs/bevformer

cd /home/mduric/projects/BEVFormer

export PYTHONPATH=$PWD:$PWD/tools:$PYTHONPATH
export WANDB_DIR=/scratch/izar/mduric/wandb
export WANDB_MODE=online

CONFIG=/home/mduric/projects/BEVFormer/projects/configs/bevformer/bevformer_base.py
WORKDIR=/scratch/izar/mduric/work_dirs/bevformer_base_bev200_baseline

mkdir -p /scratch/izar/mduric/wandb
mkdir -p /scratch/izar/mduric/slurm_logs
mkdir -p $WORKDIR

torchrun --nproc_per_node=4 tools/train.py \
  $CONFIG \
  --launcher pytorch \
  --work-dir $WORKDIR \
  2>&1 | tee $WORKDIR/train_${SLURM_JOB_ID}.log