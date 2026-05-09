#!/bin/bash
#SBATCH --job-name=bev_base_map
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/izar/tlphan/slurm_logs/%x_%j.out

source /home/tlphan/miniconda3/etc/profile.d/conda.sh
conda activate /home/tlphan/miniconda3/envs/bev

cd /home/tlphan/cs503/BEVFormer

export PYTHONPATH=$PWD:$PWD/tools:$PYTHONPATH

CONFIG=/home/tlphan/cs503/BEVFormer/projects/configs/bevformer/bevformer_base_map.py
WORKDIR=/scratch/izar/tlphan/work_dirs/bevformer_base_map

mkdir -p /scratch/izar/tlphan/slurm_logs
mkdir -p $WORKDIR

torchrun --nproc_per_node=4 tools/train.py \
  $CONFIG \
  --launcher pytorch \
  --work-dir $WORKDIR \
  2>&1 | tee $WORKDIR/train_${SLURM_JOB_ID}.log
