import os
import argparse
import random
import tempfile
import mmcv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample


def build_gt_boxes(nusc, sample_token):
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
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_path = nusc.get_sample_data_path(lidar_token)
    return os.path.exists(lidar_path), lidar_path


def render_one_model(nusc, pred_data, sample_token, out_path, score_thr):
    gt_boxes = build_gt_boxes(nusc, sample_token)
    pred_boxes = build_pred_boxes(pred_data, sample_token, score_thr)

    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()

    gt_annotations.add_boxes(sample_token, gt_boxes)
    pred_annotations.add_boxes(sample_token, pred_boxes)

    visualize_sample(
        nusc,
        sample_token,
        gt_annotations,
        pred_annotations,
        savepath=out_path,
    )

    plt.close("all")

    return len(gt_boxes), len(pred_boxes)


def add_title(img, title, height=55):
    canvas = Image.new("RGB", (img.width, img.height + height), "white")
    canvas.paste(img, (0, height))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    # Center text.
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    x = max((img.width - text_w) // 2, 5)

    draw.text((x, 12), title, fill="black", font=font)

    return canvas


def make_side_by_side(left_path, right_path, out_path, left_title, right_title):
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    # Resize to same height.
    target_h = min(left.height, right.height)
    left_w = int(left.width * target_h / left.height)
    right_w = int(right.width * target_h / right.height)

    left = left.resize((left_w, target_h), Image.BICUBIC)
    right = right.resize((right_w, target_h), Image.BICUBIC)

    left = add_title(left, left_title)
    right = add_title(right, right_title)

    gap = 20
    canvas = Image.new(
        "RGB",
        (left.width + gap + right.width, max(left.height, right.height)),
        "white",
    )

    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))

    canvas.save(out_path)


def choose_tokens(common_tokens, num_samples, start_idx, random_samples, seed):
    tokens = list(common_tokens)

    if random_samples:
        rng = random.Random(seed)
        rng.shuffle(tokens)
        return tokens

    return tokens[start_idx:]


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side BEV visualization: ours vs BEVFormer."
    )

    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--ours-json", required=True)
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")

    parser.add_argument("--ours-name", default="Ours")
    parser.add_argument("--base-name", default="BEVFormer-base")

    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.2)

    parser.add_argument("--random-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--keep-individual",
        action="store_true",
        help="Keep individual rendered images in addition to side-by-side comparison.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading nuScenes...")
    nusc = NuScenes(
        version=args.version,
        dataroot=args.dataroot,
        verbose=True,
    )

    print(f"[INFO] Loading ours JSON: {args.ours_json}")
    ours_data = mmcv.load(args.ours_json)

    print(f"[INFO] Loading baseline JSON: {args.base_json}")
    base_data = mmcv.load(args.base_json)

    ours_tokens = set(ours_data["results"].keys())
    base_tokens = set(base_data["results"].keys())
    common_tokens = sorted(list(ours_tokens & base_tokens))

    print(f"[INFO] Ours tokens: {len(ours_tokens)}")
    print(f"[INFO] Base tokens: {len(base_tokens)}")
    print(f"[INFO] Common tokens: {len(common_tokens)}")

    candidate_tokens = choose_tokens(
        common_tokens=common_tokens,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        random_samples=args.random_samples,
        seed=args.seed,
    )

    rendered = 0

    tmp_dir = os.path.join(args.out_dir, "_tmp_single")
    os.makedirs(tmp_dir, exist_ok=True)

    for idx, sample_token in enumerate(candidate_tokens):
        if rendered >= args.num_samples:
            break

        exists, lidar_path = lidar_file_exists(nusc, sample_token)
        if not exists:
            print(f"[SKIP] Missing LiDAR: {lidar_path}")
            continue

        short_token = sample_token[:8]

        ours_tmp = os.path.join(tmp_dir, f"{rendered:04d}_{short_token}_ours.png")
        base_tmp = os.path.join(tmp_dir, f"{rendered:04d}_{short_token}_base.png")

        compare_out = os.path.join(
            args.out_dir,
            f"sample_{rendered:04d}_{short_token}_compare.png",
        )

        try:
            gt_n, ours_pred_n = render_one_model(
                nusc=nusc,
                pred_data=ours_data,
                sample_token=sample_token,
                out_path=ours_tmp,
                score_thr=args.score_thr,
            )

            _, base_pred_n = render_one_model(
                nusc=nusc,
                pred_data=base_data,
                sample_token=sample_token,
                out_path=base_tmp,
                score_thr=args.score_thr,
            )

            left_title = f"{args.ours_name} | pred>{args.score_thr} ({ours_pred_n} boxes)"
            right_title = f"{args.base_name} | pred>{args.score_thr} ({base_pred_n} boxes)"

            make_side_by_side(
                left_path=ours_tmp,
                right_path=base_tmp,
                out_path=compare_out,
                left_title=left_title,
                right_title=right_title,
            )

            print(
                f"[SAVED] {compare_out} | "
                f"GT={gt_n}, ours_pred={ours_pred_n}, base_pred={base_pred_n}"
            )

            rendered += 1

        except Exception as e:
            print(f"[SKIP] Failed sample {sample_token}: {e}")
            continue

    if not args.keep_individual:
        # Remove individual temporary images.
        try:
            for fname in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, fname))
            os.rmdir(tmp_dir)
        except Exception:
            pass

    print(f"[DONE] Rendered {rendered} side-by-side comparisons.")

    if rendered == 0:
        print("[WARNING] No samples rendered. Check LiDAR files and common tokens.")


if __name__ == "__main__":
    main()