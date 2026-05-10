#!/bin/bash
# Collect results from eval_all_epochs.sh jobs.
# Usage: bash collect_eval_results.sh
#
# Log output format (from tools/test.py line: print(dataset.evaluate(...))):
#
#   Human-readable block (from _evaluate_seg):
#     Map Segmentation mIoU: 0.3500
#       drivable_area       : 0.5200
#       ped_crossing        : 0.1200
#       ...
#
#   Final printed dict contains keys like:
#     'pts_bbox_NuScenes/NDS': 0.0386
#     'pts_bbox_NuScenes/mAP': 0.0016
#     'seg/mIoU': 0.3500
#     'seg/drivable_area_IoU': 0.5200
#     ...

WORKDIR=/scratch/izar/tlphan/work_dirs/bevformer_small_map

# helper: extract value after a key in dict-printed line
# usage: extract_key LOG "key_string"
extract_key() {
    grep -oP "(?<='$2': )[0-9]+\.[0-9]+" "$1" | head -1
}

printf "%-6s  %-7s %-7s | %-9s %-9s %-9s %-9s %-9s %-9s | %-7s\n" \
    "Epoch" "mAP" "NDS" "drivable" "ped_xing" "walkway" "stop_ln" "carpark" "divider" "mIoU"
printf '%0.s-' {1..95}; echo

for epoch in $(seq 1 21); do
    LOG=$WORKDIR/eval_epoch_${epoch}/eval.log

    if [ ! -f "$LOG" ]; then
        printf "%-6s  (not done yet)\n" "$epoch"
        continue
    fi

    # Detection metrics — from the printed result dict
    mAP=$(extract_key "$LOG" "pts_bbox_NuScenes/mAP")
    NDS=$(extract_key "$LOG" "pts_bbox_NuScenes/NDS")

    # Segmentation metrics — from the human-readable _evaluate_seg block
    # Format: "  drivable_area       : 0.4500"
    driv=$(grep "drivable_area" "$LOG" | grep -oP "(?<=: )[0-9]+\.[0-9]+" | head -1)
    ped=$( grep "ped_crossing"  "$LOG" | grep -oP "(?<=: )[0-9]+\.[0-9]+" | head -1)
    walk=$(grep "walkway"       "$LOG" | grep -oP "(?<=: )[0-9]+\.[0-9]+" | head -1)
    stop=$(grep "stop_line"     "$LOG" | grep -oP "(?<=: )[0-9]+\.[0-9]+" | head -1)
    cark=$(grep "carpark_area"  "$LOG" | grep -oP "(?<=: )[0-9]+\.[0-9]+" | head -1)
    divr=$(grep "divider"       "$LOG" | grep -oP "(?<=: )[0-9]+\.[0-9]+" | head -1)
    miou=$(grep "Map Segmentation mIoU" "$LOG" | grep -oP "[0-9]+\.[0-9]+" | head -1)

    printf "%-6s  %-7s %-7s | %-9s %-9s %-9s %-9s %-9s %-9s | %-7s\n" \
        "$epoch" \
        "${mAP:--}" "${NDS:--}" \
        "${driv:--}" "${ped:--}" "${walk:--}" "${stop:--}" "${cark:--}" "${divr:--}" \
        "${miou:--}"
done
