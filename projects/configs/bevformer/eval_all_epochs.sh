#!/bin/bash
#SBATCH --job-name=bev_eval
#SBATCH --time=01:30:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-21
#SBATCH --output=/home/tlphan/cs503/BEVFormer/slurm_logs/bev_eval_%A_%a.out
#SBATCH --error=/home/tlphan/cs503/BEVFormer/slurm_logs/bev_eval_%A_%a.err

source /home/tlphan/miniconda3/etc/profile.d/conda.sh
conda activate /home/tlphan/miniconda3/envs/bev

cd /home/tlphan/cs503/BEVFormer
export PYTHONPATH=$PWD:$PWD/tools:$PYTHONPATH

EPOCH=${SLURM_ARRAY_TASK_ID}
CONFIG=/home/tlphan/cs503/BEVFormer/projects/configs/bevformer/bevformer_small_map.py
CKPT=/scratch/izar/tlphan/work_dirs/bevformer_small_map/epoch_${EPOCH}.pth
OUTDIR=/scratch/izar/tlphan/work_dirs/bevformer_small_map/eval_epoch_${EPOCH}

mkdir -p $OUTDIR
mkdir -p /home/tlphan/cs503/BEVFormer/slurm_logs

echo "=== Evaluating epoch ${EPOCH} (detection + segmentation) ==="
echo "Checkpoint: $CKPT"

# --eval bbox triggers CustomNuScenesDataset.evaluate(), which automatically
# appends per-class seg IoU whenever seg_preds are present in model outputs.
python tools/test.py \
    $CONFIG \
    $CKPT \
    --eval bbox \
    --tmpdir $OUTDIR \
    2>&1 | tee $OUTDIR/eval.log

echo "=== Done epoch ${EPOCH} ==="
