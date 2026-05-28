#!/bin/bash
# Run segmentation + detection eval on every 3rd epoch checkpoint.
# Usage (interactive, after srun into a GPU node):
#   bash projects/configs/bevformer/run_multi_epoch_eval.sh [WORKDIR] [STEP]
#
# Or as a single-GPU sbatch:
#   sbatch projects/configs/bevformer/run_multi_epoch_eval.sh
#
#SBATCH --job-name=bev_multi_eval
#SBATCH --time=08:00:00
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

WORKDIR=${1:-/scratch/izar/tlphan/work_dirs/vjepa_temporal_motion_2026_05_21}
STEP=${2:-5}          # evaluate every STEP epochs
TOTAL_EPOCHS=30
CONFIG=projects/configs/bevformer/bevformer_small_map.py
RESULTS_CSV=$WORKDIR/eval_results.csv

mkdir -p $WORKDIR/viz

# CSV header
echo "epoch,mAP,NDS,seg_mIoU,drivable_area,ped_crossing,walkway,stop_line,carpark_area,divider" > $RESULTS_CSV

for EPOCH in $(seq $TOTAL_EPOCHS -$STEP $STEP); do
    CKPT=$WORKDIR/epoch_${EPOCH}.pth
    if [ ! -f "$CKPT" ]; then
        echo "Skipping epoch $EPOCH — checkpoint not found: $CKPT"
        continue
    fi

    VIZ_DIR=$WORKDIR/viz/epoch_${EPOCH}
    LOG=$WORKDIR/eval_epoch${EPOCH}.log

    echo "========================================"
    echo "Evaluating epoch $EPOCH ..."
    echo "========================================"

    torchrun --nproc_per_node=1 tools/test.py \
        $CONFIG \
        $CKPT \
        --eval bbox \
        --launcher pytorch \
        --eval-options "seg_viz_dir=$VIZ_DIR" "seg_num_viz=16" \
        2>&1 | tee $LOG

    # Parse mAP, NDS from log
    MAP=$(grep -oP "mAP: \K[0-9.]+" $LOG | tail -1)
    NDS=$(grep -oP "NDS: \K[0-9.]+" $LOG | tail -1)

    # Parse seg IoUs from log
    SEG_MIOU=$(grep -oP "Map Segmentation mIoU: \K[0-9.]+" $LOG | tail -1)
    DA=$(grep -oP "drivable_area\s+: \K[0-9.]+" $LOG | tail -1)
    PC=$(grep -oP "ped_crossing\s+: \K[0-9.]+" $LOG | tail -1)
    WW=$(grep -oP "walkway\s+: \K[0-9.]+" $LOG | tail -1)
    SL=$(grep -oP "stop_line\s+: \K[0-9.]+" $LOG | tail -1)
    CA=$(grep -oP "carpark_area\s+: \K[0-9.]+" $LOG | tail -1)
    DV=$(grep -oP "divider\s+: \K[0-9.]+" $LOG | tail -1)

    echo "$EPOCH,$MAP,$NDS,$SEG_MIOU,$DA,$PC,$WW,$SL,$CA,$DV" >> $RESULTS_CSV
    echo "Epoch $EPOCH done → mAP=$MAP  NDS=$NDS  seg_mIoU=$SEG_MIOU"
done

echo ""
echo "========================================"
echo "All done. Results summary:"
echo "========================================"
cat $RESULTS_CSV
