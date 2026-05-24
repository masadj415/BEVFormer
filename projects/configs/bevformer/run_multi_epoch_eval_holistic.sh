#!/bin/bash
# Holistic eval: detection + segmentation + ego trajectory + agent motion.
# Usage (interactive):
#   bash projects/configs/bevformer/run_multi_epoch_eval_holistic.sh [WORKDIR] [STEP]
#
# Or as sbatch:
#   sbatch projects/configs/bevformer/run_multi_epoch_eval_holistic.sh
#
#SBATCH --job-name=bev_holistic_eval
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

WORKDIR=${1:-/scratch/izar/tlphan/work_dirs/bevformer_holistic}
STEP=${2:-5}          # evaluate every STEP epochs
TOTAL_EPOCHS=30
CONFIG=${3:-projects/configs/bevformer/bevformer_vjepa_temporal_izar.py}
RESULTS_CSV=$WORKDIR/eval_results_holistic.csv

mkdir -p $WORKDIR/viz

# CSV header
echo "epoch,mAP,NDS,seg_mIoU,drivable_area,ped_crossing,walkway,stop_line,carpark_area,divider,ego_ADE,ego_FDE,motion_ADE,motion_FDE" > $RESULTS_CSV

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

    # Parse ego trajectory and agent motion ADE/FDE from log
    EGO_ADE=$(grep -oP "Ego trajectory  ADE: \K[0-9.]+" $LOG | tail -1)
    EGO_FDE=$(grep -oP "Ego trajectory  ADE: [0-9.]+ m   FDE: \K[0-9.]+" $LOG | tail -1)
    MOT_ADE=$(grep -oP "Agent motion    ADE: \K[0-9.]+" $LOG | tail -1)
    MOT_FDE=$(grep -oP "Agent motion    ADE: [0-9.]+ m   FDE: \K[0-9.]+" $LOG | tail -1)

    echo "$EPOCH,$MAP,$NDS,$SEG_MIOU,$DA,$PC,$WW,$SL,$CA,$DV,$EGO_ADE,$EGO_FDE,$MOT_ADE,$MOT_FDE" >> $RESULTS_CSV
    echo "Epoch $EPOCH done → mAP=$MAP  NDS=$NDS  seg_mIoU=$SEG_MIOU  ego_ADE=$EGO_ADE  ego_FDE=$EGO_FDE  mot_ADE=$MOT_ADE  mot_FDE=$MOT_FDE"
done

echo ""
echo "========================================"
echo "All done. Results summary:"
echo "========================================"
cat $RESULTS_CSV
