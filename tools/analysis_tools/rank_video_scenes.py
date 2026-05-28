import argparse
import math
import mmcv
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name


def dist_xy(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_gt_boxes(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    gt = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_name = category_to_detection_name(ann["category_name"])
        if det_name is None:
            continue
        gt.append({
            "name": det_name,
            "translation": ann["translation"],
        })

    return gt


def match_preds_to_gt(preds, gt, score_thr=0.35, dist_thr=2.0):
    preds = [p for p in preds if float(p.get("detection_score", 0.0)) >= score_thr]

    matched_gt = set()
    tp = 0

    for p in sorted(preds, key=lambda x: float(x.get("detection_score", 0.0)), reverse=True):
        best_j = None
        best_d = 1e9

        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            if p["detection_name"] != g["name"]:
                continue

            d = dist_xy(p["translation"], g["translation"])
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is not None and best_d <= dist_thr:
            tp += 1
            matched_gt.add(best_j)

    fp = len(preds) - tp
    fn = len(gt) - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return tp, fp, fn, precision, recall, len(preds), len(gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--score-thr", type=float, default=0.35)
    parser.add_argument("--dist-thr", type=float, default=2.0)
    parser.add_argument("--min-frames", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    pred_data = mmcv.load(args.result_json)
    pred_tokens = set(pred_data["results"].keys())

    rows = []

    for scene_idx, scene in enumerate(nusc.scene):
        token = scene["first_sample_token"]
        scene_tokens = []

        while token:
            if token in pred_tokens:
                scene_tokens.append(token)
            sample = nusc.get("sample", token)
            token = sample["next"]

        if len(scene_tokens) < args.min_frames:
            continue

        total_tp = total_fp = total_fn = 0
        total_preds = total_gt = 0

        for sample_token in scene_tokens[:args.min_frames]:
            preds = pred_data["results"].get(sample_token, [])
            gt = get_gt_boxes(nusc, sample_token)

            tp, fp, fn, precision, recall, n_preds, n_gt = match_preds_to_gt(
                preds=preds,
                gt=gt,
                score_thr=args.score_thr,
                dist_thr=args.dist_thr,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_preds += n_preds
            total_gt += n_gt

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # Prefer accurate but also visually rich scenes.
        richness = min(total_preds / (args.min_frames * 8), 1.0)
        video_score = 0.75 * f1 + 0.25 * richness

        rows.append({
            "scene_idx": scene_idx,
            "scene_name": scene["name"],
            "frames": len(scene_tokens),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "preds": total_preds,
            "gt": total_gt,
            "video_score": video_score,
        })

    rows = sorted(rows, key=lambda r: r["video_score"], reverse=True)

    print(f"Top {args.top_k} scenes for video:")
    print("scene_idx | scene_name | frames | precision | recall | f1 | preds | gt | video_score")
    print("-" * 100)

    for r in rows[:args.top_k]:
        print(
            f"{r['scene_idx']:9d} | {r['scene_name']:12s} | "
            f"{r['frames']:6d} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['preds']:5d} | {r['gt']:5d} | {r['video_score']:.3f}"
        )


if __name__ == "__main__":
    main()