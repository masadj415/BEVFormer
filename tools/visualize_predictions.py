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


def draw_box_bev(ax, box, color, linestyle="-", linewidth=1.8, text=None):
    corners = box.corners()

    # Bottom face of the 3D box in BEV.
    x = corners[0, [0, 1, 2, 3, 0]]
    y = corners[1, [0, 1, 2, 3, 0]]

    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)

    if text is not None:
        ax.text(box.center[0], box.center[1], text, fontsize=6, color=color)


def render_bev(nusc, sample_token, pred_data, out_path, score_thr):
    """
    BEV visualization WITHOUT loading LiDAR point cloud.
    Green = ground truth.
    Red dashed = predictions.
    """
    sample = nusc.get("sample", sample_token)

    # We use LIDAR_TOP metadata only to get ego pose.
    # This does NOT load the actual .pcd.bin LiDAR file.
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    pose_record = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    fig, ax = plt.subplots(figsize=(10, 10))

    # -------------------------
    # Ground-truth boxes
    # -------------------------
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_name = category_to_detection_name(ann["category_name"])

        if det_name is None:
            continue

        box = Box(
            center=ann["translation"],
            size=ann["size"],
            orientation=Quaternion(ann["rotation"]),
            name=det_name,
            token=ann_token,
        )

        # global -> ego frame
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        draw_box_bev(
            ax,
            box,
            color="green",
            linestyle="-",
            linewidth=1.8,
            text=f"GT {det_name}",
        )

    # -------------------------
    # Predicted boxes
    # -------------------------
    pred_records = pred_data["results"][sample_token]

    for record in pred_records:
        score = float(record.get("detection_score", 0.0))

        if score < score_thr:
            continue

        det_name = record["detection_name"]

        box = Box(
            center=record["translation"],
            size=record["size"],
            orientation=Quaternion(record["rotation"]),
            name=det_name,
            token="predicted",
        )

        # global -> ego frame
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        draw_box_bev(
            ax,
            box,
            color="red",
            linestyle="--",
            linewidth=1.4,
            text=f"P {det_name} {score:.2f}",
        )

    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_aspect("equal")
    ax.grid(True)

    ax.set_xlabel("x ego [m]")
    ax.set_ylabel("y ego [m]")
    ax.set_title(
        f"BEV without LiDAR points | green=GT, red=prediction | score>{score_thr}"
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
