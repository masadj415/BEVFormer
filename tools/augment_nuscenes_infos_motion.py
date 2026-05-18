import argparse
import warnings
import numpy as np
import mmcv

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion


def global_to_lidar(point_global, info):
    point = np.array(point_global, dtype=np.float64)

    # global -> ego
    point = point - np.array(info["ego2global_translation"], dtype=np.float64)
    point = Quaternion(info["ego2global_rotation"]).inverse.rotate(point)

    # ego -> lidar
    point = point - np.array(info["lidar2ego_translation"], dtype=np.float64)
    point = Quaternion(info["lidar2ego_rotation"]).inverse.rotate(point)

    return np.asarray(point, dtype=np.float32)


def rotate_to_local_xy(offset_xy, yaw):
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    x, y = offset_xy
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def build_candidates(nusc, info):
    sample = nusc.get("sample", info["token"])
    candidates = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det_name = category_to_detection_name(ann["category_name"])

        if det_name is None:
            continue

        center_lidar = global_to_lidar(ann["translation"], info)

        candidates.append(
            {
                "ann_token": ann_token,
                "instance_token": ann["instance_token"],
                "name": det_name,
                "center_lidar": center_lidar,
                "ann": ann,
            }
        )

    return candidates


def match_info_boxes_to_nusc_annotations(nusc, info, max_dist=1.0):
    """
    Match existing info['gt_boxes'] to nuScenes sample annotations.
    We match by class and nearest center in current lidar coordinates.
    """
    gt_boxes = np.asarray(info["gt_boxes"])
    gt_names = np.asarray(info["gt_names"])
    candidates = build_candidates(nusc, info)

    used = set()
    ann_tokens = []
    instance_tokens = []
    matched_anns = []

    for i in range(len(gt_boxes)):
        gt_name = str(gt_names[i])
        gt_center = gt_boxes[i, :3]

        best_j = None
        best_dist = float("inf")

        for j, cand in enumerate(candidates):
            if j in used:
                continue
            if cand["name"] != gt_name:
                continue

            dist = np.linalg.norm(cand["center_lidar"][:2] - gt_center[:2])

            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is None or best_dist > max_dist:
            warnings.warn(
                f"Could not confidently match GT box {i}, "
                f"class={gt_name}, best_dist={best_dist:.3f}, sample={info['token']}"
            )
            ann_tokens.append("")
            instance_tokens.append("")
            matched_anns.append(None)
        else:
            used.add(best_j)
            cand = candidates[best_j]
            ann_tokens.append(cand["ann_token"])
            instance_tokens.append(cand["instance_token"])
            matched_anns.append(cand["ann"])

    return ann_tokens, instance_tokens, matched_anns


def compute_future_traj(nusc, info, matched_anns, future_steps=6, local_coords=False):
    gt_boxes = np.asarray(info["gt_boxes"])
    num_gt = len(gt_boxes)

    gt_fut_traj = np.zeros((num_gt, future_steps, 2), dtype=np.float32)
    gt_fut_traj_mask = np.zeros((num_gt, future_steps), dtype=np.float32)

    for i, ann in enumerate(matched_anns):
        if ann is None:
            continue

        cur_center = global_to_lidar(ann["translation"], info)
        cur_yaw = float(gt_boxes[i, 6]) if gt_boxes.shape[1] > 6 else 0.0

        next_ann_token = ann["next"]

        for k in range(future_steps):
            if next_ann_token == "":
                break

            fut_ann = nusc.get("sample_annotation", next_ann_token)
            fut_center = global_to_lidar(fut_ann["translation"], info)

            offset_xy = fut_center[:2] - cur_center[:2]

            if local_coords:
                offset_xy = rotate_to_local_xy(offset_xy, cur_yaw)

            gt_fut_traj[i, k] = offset_xy
            gt_fut_traj_mask[i, k] = 1.0

            next_ann_token = fut_ann["next"]

    return gt_fut_traj, gt_fut_traj_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--in-pkl", required=True)
    parser.add_argument("--out-pkl", required=True)
    parser.add_argument("--future-steps", type=int, default=6)
    parser.add_argument("--local-coords", action="store_true")
    parser.add_argument("--max-match-dist", type=float, default=1.0)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    data = mmcv.load(args.in_pkl)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data

    total_boxes = 0
    total_unmatched = 0
    total_future_valid = 0

    for idx, info in enumerate(mmcv.track_iter_progress(infos)):
        ann_tokens, instance_tokens, matched_anns = match_info_boxes_to_nusc_annotations(
            nusc,
            info,
            max_dist=args.max_match_dist,
        )

        gt_fut_traj, gt_fut_traj_mask = compute_future_traj(
            nusc,
            info,
            matched_anns,
            future_steps=args.future_steps,
            local_coords=args.local_coords,
        )

        info["gt_ann_tokens"] = ann_tokens
        info["gt_instance_tokens"] = instance_tokens
        info["gt_fut_traj"] = gt_fut_traj
        info["gt_fut_traj_mask"] = gt_fut_traj_mask

        total_boxes += len(ann_tokens)
        total_unmatched += sum(1 for x in ann_tokens if x == "")
        total_future_valid += int(gt_fut_traj_mask.sum())

        if idx < 3:
            print("\nExample sample:", info["token"])
            print("gt_boxes:", np.asarray(info["gt_boxes"]).shape)
            print("gt_fut_traj:", gt_fut_traj.shape)
            print("valid future steps:", int(gt_fut_traj_mask.sum()))

    mmcv.dump(data, args.out_pkl)

    print("\nSaved:", args.out_pkl)
    print("Total GT boxes:", total_boxes)
    print("Unmatched boxes:", total_unmatched)
    print("Valid future trajectory steps:", total_future_valid)


if __name__ == "__main__":
    main()