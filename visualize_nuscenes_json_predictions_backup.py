import os
import argparse
import mmcv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample


CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def get_color(nusc, category_name):
    if category_name == "bicycle":
        return nusc.colormap["vehicle.bicycle"]
    if category_name == "construction_vehicle":
        return nusc.colormap["vehicle.construction"]
    if category_name == "traffic_cone":
        return nusc.colormap["movable_object.trafficcone"]

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]

    return [255, 0, 0]


def make_pred_boxes_for_camera(pred_records, score_thr):
    boxes = []
    for record in pred_records:
        score = float(record.get("detection_score", 0.0))
        if score < score_thr:
            continue

        box = Box(
            center=record["translation"],
            size=record["size"],
            orientation=Quaternion(record["rotation"]),
            name=record["detection_name"],
            token="predicted",
        )
        boxes.append((box, score))
    return boxes


def get_predicted_data(nusc, sample_data_token, pred_records, score_thr, box_vis_level=BoxVisibility.ANY):
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        imsize = (sd_record["width"], sd_record["height"])
    else:
        cam_intrinsic = None
        imsize = None

    pred_boxes_with_scores = make_pred_boxes_for_camera(pred_records, score_thr)

    box_list = []
    score_list = []

    for box, score in pred_boxes_with_scores:
        # global -> ego
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        # ego -> sensor
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        if sensor_record["modality"] == "camera":
            if not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

        box_list.append(box)
        score_list.append(score)

    return data_path, box_list, score_list, cam_intrinsic


def render_bev(nusc, sample_token, pred_data, out_path, score_thr):
    gt_boxes = []
    pred_boxes = []

    sample = nusc.get("sample", sample_token)

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_name = category_to_detection_name(ann["category_name"])
        if det_name is None:
            continue

        velocity = nusc.box_velocity(ann_token)[:2]
        if np.any(np.isnan(velocity)):
            velocity = (0.0, 0.0)

        gt_boxes.append(
            DetectionBox(
                sample_token=ann["sample_token"],
                translation=tuple(ann["translation"]),
                size=tuple(ann["size"]),
                rotation=tuple(ann["rotation"]),
                velocity=tuple(velocity),
                ego_translation=(0.0, 0.0, 0.0),
                num_pts=int(ann.get("num_lidar_pts", -1)),
                detection_name=det_name,
                detection_score=-1.0,
                attribute_name="",
            )
        )

    for record in pred_data["results"][sample_token]:
        score = float(record.get("detection_score", 0.0))
        if score < score_thr:
            continue

        pred_boxes.append(
            DetectionBox(
                sample_token=record["sample_token"],
                translation=tuple(record["translation"]),
                size=tuple(record["size"]),
                rotation=tuple(record["rotation"]),
                velocity=tuple(record.get("velocity", (0.0, 0.0))),
                ego_translation=tuple(record.get("ego_translation", (0.0, 0.0, 0.0))),
                num_pts=int(record.get("num_pts", -1)),
                detection_name=record["detection_name"],
                detection_score=score,
                attribute_name=record.get("attribute_name", ""),
            )
        )

    gt_eval = EvalBoxes()
    pred_eval = EvalBoxes()
    gt_eval.add_boxes(sample_token, gt_boxes)
    pred_eval.add_boxes(sample_token, pred_boxes)

    visualize_sample(
        nusc,
        sample_token,
        gt_eval,
        pred_eval,
        savepath=out_path,
    )


def render_camera_grid(nusc, sample_token, pred_data, out_path, score_thr):
    sample = nusc.get("sample", sample_token)
    pred_records = pred_data["results"][sample_token]

    fig, ax = plt.subplots(4, 3, figsize=(24, 18))

    for cam_idx, cam in enumerate(CAMS):
        row_pred = 0 if cam_idx < 3 else 1
        row_gt = row_pred + 2
        col = cam_idx % 3

        sample_data_token = sample["data"][cam]
        sd_record = nusc.get("sample_data", sample_data_token)

        data_path, pred_boxes, pred_scores, camera_intrinsic = get_predicted_data(
            nusc,
            sample_data_token,
            pred_records,
            score_thr=score_thr,
            box_vis_level=BoxVisibility.ANY,
        )

        _, gt_boxes, _ = nusc.get_sample_data(
            sample_data_token,
            box_vis_level=BoxVisibility.ANY,
        )

        img = Image.open(data_path)

        ax[row_pred, col].imshow(img)
        ax[row_gt, col].imshow(img)

        for box in pred_boxes:
            c = np.array(get_color(nusc, box.name)) / 255.0
            box.render(ax[row_pred, col], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        for box in gt_boxes:
            c = np.array(get_color(nusc, box.name)) / 255.0
            box.render(ax[row_gt, col], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        for r in [row_pred, row_gt]:
            ax[r, col].set_xlim(0, img.size[0])
            ax[r, col].set_ylim(img.size[1], 0)
            ax[r, col].axis("off")
            ax[r, col].set_aspect("equal")

        ax[row_pred, col].set_title(f"PRED {sd_record['channel']} | score>{score_thr}")
        ax[row_gt, col].set_title(f"GT {sd_record['channel']}")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading nuScenes...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    print(f"[INFO] Loading predictions: {args.result_json}")
    pred_data = mmcv.load(args.result_json)

    sample_tokens = list(pred_data["results"].keys())
    end_idx = min(args.start_idx + args.num_samples, len(sample_tokens))

    print(f"[INFO] Total predicted samples: {len(sample_tokens)}")
    print(f"[INFO] Visualizing samples {args.start_idx} to {end_idx - 1}")

    for i in range(args.start_idx, end_idx):
        sample_token = sample_tokens[i]
        print(f"[INFO] Rendering {i}: {sample_token}")

        camera_out = os.path.join(args.out_dir, f"sample_{i:04d}_camera.png")
        bev_out = os.path.join(args.out_dir, f"sample_{i:04d}_bev.png")

        render_camera_grid(nusc, sample_token, pred_data, out_path=camera_out, score_thr=args.score_thr)
        render_bev(nusc, sample_token, pred_data, out_path=bev_out, score_thr=args.score_thr)

        print(f"[SAVED] {camera_out}")
        print(f"[SAVED] {bev_out}")

    print("[DONE]")


if __name__ == "__main__":
    main()
