#!/bin/bash
#SBATCH --job-name=bev_seg_eval
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/tlphan/cs503/BEVFormer/slurm_logs/%x_%j.out
#SBATCH --error=/home/tlphan/cs503/BEVFormer/slurm_logs/%x_%j.err

source /home/tlphan/miniconda3/etc/profile.d/conda.sh
conda activate /home/tlphan/miniconda3/envs/bev

cd /home/tlphan/cs503/BEVFormer
export PYTHONPATH=$PWD:$PWD/tools:$PYTHONPATH

CONFIG=/home/tlphan/cs503/BEVFormer/projects/configs/bevformer/bevformer_small_map.py
CHECKPOINT=${1:-/scratch/izar/tlphan/work_dirs/bevformer_small_map/latest.pth}
OUTDIR=/scratch/izar/tlphan/work_dirs/bevformer_small_map/seg_eval

mkdir -p $OUTDIR

echo "Evaluating checkpoint: $CHECKPOINT"

python tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --eval bbox \
    --tmpdir $OUTDIR \
    2>&1 | tee $OUTDIR/seg_eval_${SLURM_JOB_ID}.log
