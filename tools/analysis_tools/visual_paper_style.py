import os
import argparse
import random
import mmcv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample


def build_gt_boxes(nusc, sample_token):
    """Build ground-truth DetectionBox objects for one sample."""
    gt_boxes = []

    sample = nusc.get("sample", sample_token)

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_name = category_to_detection_name(ann["category_name"])

        if det_name is None:
            continue

        try:
            velocity = nusc.box_velocity(ann["token"])[:2]
            if velocity is None:
                velocity = (0.0, 0.0)
        except Exception:
            velocity = (0.0, 0.0)

        gt_boxes.append(
            DetectionBox(
                sample_token=sample_token,
                translation=tuple(ann["translation"]),
                size=tuple(ann["size"]),
                rotation=tuple(ann["rotation"]),
                velocity=tuple(velocity),
                ego_translation=(0.0, 0.0, 0.0),
                num_pts=-1,
                detection_name=det_name,
                detection_score=-1.0,
                attribute_name="",
            )
        )

    return gt_boxes


def build_pred_boxes(pred_data, sample_token, score_thr):
    """Build predicted DetectionBox objects for one sample."""
    pred_boxes = []

    if sample_token not in pred_data["results"]:
        return pred_boxes

    for pred in pred_data["results"][sample_token]:
        score = float(pred.get("detection_score", 0.0))
        if score < score_thr:
            continue

        velocity = pred.get("velocity", [0.0, 0.0])
        if velocity is None:
            velocity = [0.0, 0.0]

        pred_boxes.append(
            DetectionBox(
                sample_token=sample_token,
                translation=tuple(pred["translation"]),
                size=tuple(pred["size"]),
                rotation=tuple(pred["rotation"]),
                velocity=tuple(velocity),
                ego_translation=(0.0, 0.0, 0.0),
                num_pts=-1,
                detection_name=pred["detection_name"],
                detection_score=score,
                attribute_name=pred.get("attribute_name", ""),
            )
        )

    return pred_boxes


def lidar_file_exists(nusc, sample_token):
    """Check whether the LIDAR_TOP file for this sample exists locally."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_path = nusc.get_sample_data_path(lidar_token)
    return os.path.exists(lidar_path), lidar_path


def render_bev_paper_style(nusc, pred_data, sample_token, out_path, score_thr):
    """
    Render official nuScenes-style BEV visualization.
    Green = ground truth
    Blue = prediction
    """
    gt_boxes = build_gt_boxes(nusc, sample_token)
    pred_boxes = build_pred_boxes(pred_data, sample_token, score_thr)

    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()

    gt_annotations.add_boxes(sample_token, gt_boxes)
    pred_annotations.add_boxes(sample_token, pred_boxes)

    print(f"[INFO] Rendering sample: {sample_token}")
    print(f"[INFO] GT boxes: {len(gt_boxes)} | Pred boxes above threshold: {len(pred_boxes)}")
    print("[INFO] Colors: green = ground truth, blue = prediction")

    visualize_sample(
        nusc,
        sample_token,
        gt_annotations,
        pred_annotations,
        savepath=out_path,
    )

    plt.close("all")


def choose_sample_tokens(sample_tokens, num_samples, start_idx, random_samples, seed):
    total = len(sample_tokens)

    if random_samples:
        rng = random.Random(seed)
        k = min(num_samples, total)
        chosen = rng.sample(sample_tokens, k=k)

        print("[INFO] Random sampling enabled.")
        print(f"[INFO] Seed: {seed}")
        print(f"[INFO] Selected {len(chosen)} / {total} tokens.")

        return chosen

    end_idx = min(start_idx + num_samples, total)
    chosen = sample_tokens[start_idx:end_idx]

    print("[INFO] Sequential sampling enabled.")
    print(f"[INFO] Selected tokens from index {start_idx} to {end_idx - 1}")

    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Paper-style BEV visualization with skip-missing-LiDAR support."
    )

    parser.add_argument(
        "--dataroot",
        required=True,
        help="Path to nuScenes root, e.g. /scratch/izar/mduric/nuscenes_trainval",
    )
    parser.add_argument(
        "--result-json",
        required=True,
        help="Path to results_nusc.json",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where BEV visualizations will be saved",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes version, usually v1.0-trainval",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many BEV images you want to save",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index when using sequential sampling",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.2,
        help="Prediction score threshold",
    )
    parser.add_argument(
        "--random-samples",
        action="store_true",
        help="Randomly sample tokens instead of using sequential order",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --random-samples is enabled",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading nuScenes...")
    nusc = NuScenes(
        version=args.version,
        dataroot=args.dataroot,
        verbose=True,
    )

    print(f"[INFO] Loading prediction JSON: {args.result_json}")
    pred_data = mmcv.load(args.result_json)

    sample_tokens = list(pred_data["results"].keys())
    print(f"[INFO] Total prediction tokens: {len(sample_tokens)}")

    candidate_tokens = choose_sample_tokens(
        sample_tokens=sample_tokens,
        num_samples=len(sample_tokens) if args.random_samples else len(sample_tokens),
        start_idx=args.start_idx,
        random_samples=args.random_samples,
        seed=args.seed,
    )

    rendered = 0

    for idx, sample_token in enumerate(candidate_tokens):
        if rendered >= args.num_samples:
            break

        exists, lidar_path = lidar_file_exists(nusc, sample_token)

        if not exists:
            print(f"[SKIP] Missing LiDAR file: {lidar_path}")
            continue

        short_token = sample_token[:8]
        out_path = os.path.join(
            args.out_dir,
            f"sample_{rendered:04d}_{short_token}_bev.png",
        )

        try:
            render_bev_paper_style(
                nusc=nusc,
                pred_data=pred_data,
                sample_token=sample_token,
                out_path=out_path,
                score_thr=args.score_thr,
            )
            print(f"[SAVED] {out_path}")
            rendered += 1
        except Exception as e:
            print(f"[SKIP] Failed to render {sample_token}: {e}")

    print(f"[DONE] Successfully rendered {rendered} samples.")
    if rendered == 0:
        print("[WARNING] No samples were rendered. Most likely the needed LiDAR files are missing.")


if __name__ == "__main__":
    main()