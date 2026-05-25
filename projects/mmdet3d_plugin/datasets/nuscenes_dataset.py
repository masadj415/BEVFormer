import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(
        self,
        queue_length=4,
        bev_size=(200, 200),
        overlap_test=False,
        num_future_frames=6,
        future_dt=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.num_future_frames = num_future_frames
        self.future_dt = future_dt
        # Fast token→index lookup for following the next-frame chain
        self.token_to_idx = {
            info['token']: i for i, info in enumerate(self.data_infos)
        }
        # Precompute which samples have map data (path present in the pkl).
        # Used by MapAwareDistributedGroupSampler to guarantee ≥1 map-valid
        # sample per batch. Path presence is used as the proxy — scenes whose
        # npz files are genuinely all-zeros are still marked True here but the
        # seg loss will skip them via the per-sample validity mask (Fix 1).
        self.has_map_flags = np.array(
            [bool(info.get('maps', {}).get('map_mask')) for info in self.data_infos],
            dtype=bool,
        )

    def _compute_future_ego_waypoints(self, info):
        """
        Return future ego positions in the current ego frame.

        Follows the 'next' sample chain for num_future_frames steps.  Each
        future ego origin is expressed as (x, y) in the current ego frame.
        Waypoints for frames beyond the scene end are filled with NaN so the
        loss can mask them out.

        Returns:
            np.ndarray [num_future_frames, 2] float32
        """
        curr_trans = np.array(info['ego2global_translation'], dtype=np.float64)
        curr_R = Quaternion(info['ego2global_rotation']).rotation_matrix  # ego→global [3,3]

        waypoints = []
        cur_info = info
        for _ in range(self.num_future_frames):
            next_token = cur_info.get('next', '')
            if not next_token or next_token not in self.token_to_idx:
                break
            cur_info = self.data_infos[self.token_to_idx[next_token]]
            fut_trans = np.array(cur_info['ego2global_translation'], dtype=np.float64)
            # Vector from current ego origin to future ego origin, in global frame
            delta_global = fut_trans - curr_trans                   # [3]
            # Rotate into current ego frame: ego_pos = R^T @ delta_global
            delta_ego = curr_R.T @ delta_global                     # [3]
            waypoints.append(delta_ego[:2].astype(np.float32))

        # Pad with NaN for missing future frames
        result = np.full((self.num_future_frames, 2), float('nan'), dtype=np.float32)
        for k, wp in enumerate(waypoints):
            result[k] = wp
        return result
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)


    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

            # Load per-agent future trajectories if the pkl provides them.
            # These are already in agent-local frame (origin = agent center,
            # x-axis = agent heading direction) with shape [N, 6, 2] / [N, 6].
            # valid_flag / num_lidar_pts mask is applied here to stay in sync
            # with gt_boxes loaded by get_ann_info().
            if 'gt_fut_traj' in info:
                mask = (info['valid_flag']
                        if self.use_valid_flag
                        else info['num_lidar_pts'] > 0)
                input_dict['gt_fut_traj'] = info['gt_fut_traj'][mask]
                input_dict['gt_fut_traj_mask'] = info['gt_fut_traj_mask'][mask]

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        if not self.test_mode:
            input_dict['gt_future_ego'] = self._compute_future_ego_waypoints(info)

        # Required by LoadMapMaskFromNpz — pass the map npz path stored in the pkl.
        input_dict['maps'] = info.get('maps', {})

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def evaluate(self, results, metric='bbox', logger=None,
                 jsonfile_prefix=None, result_names=['pts_bbox'],
                 show=False, out_dir=None, pipeline=None):
        results_dict = super().evaluate(
            results, metric=metric, logger=logger,
            jsonfile_prefix=jsonfile_prefix, result_names=result_names,
            show=show, out_dir=out_dir, pipeline=pipeline,
        )

        # ego/motion keys live inside pts_bbox (set by simple_test_pts)
        _sample = results[0].get('pts_bbox', results[0]) if results else {}
        if results and 'ego_waypoints' in _sample:
            ego_ade, ego_fde = self._eval_ego_trajectory(results)
            results_dict['ego/ADE'] = ego_ade
            results_dict['ego/FDE'] = ego_fde
            mmcv.utils.print_log(
                f'Ego trajectory  ADE: {ego_ade:.4f} m   FDE: {ego_fde:.4f} m',
                logger=logger,
            )

        if results and 'motion_preds' in _sample:
            motion_ade, motion_fde = self._eval_agent_motion(results)
            results_dict['motion/ADE'] = motion_ade
            results_dict['motion/FDE'] = motion_fde
            mmcv.utils.print_log(
                f'Agent motion    ADE: {motion_ade:.4f} m   FDE: {motion_fde:.4f} m',
                logger=logger,
            )

        return results_dict

    def _eval_ego_trajectory(self, results):
        """Mean ADE and FDE for ego future waypoints (meters)."""
        ades, fdes = [], []
        for i, result in enumerate(results):
            r = result.get('pts_bbox', result)
            if 'ego_waypoints' not in r:
                continue
            gt = self._compute_future_ego_waypoints(self.data_infos[i])  # [6, 2]
            pred = r['ego_waypoints']                                      # [6, 2]
            valid = np.isfinite(gt).all(axis=-1)                          # [6] bool
            if not valid.any():
                continue
            err = np.linalg.norm(pred[valid] - gt[valid], axis=-1)        # [N_valid]
            ades.append(err.mean())
            fdes.append(err[-1])
        if not ades:
            return float('nan'), float('nan')
        return float(np.mean(ades)), float(np.mean(fdes))

    def _eval_agent_motion(self, results, match_thresh=4.0, min_speed=0.5):
        """
        ADE / FDE for moving agents (m, agent-local frame).

        Matches each GT moving agent to the nearest predicted query center
        within match_thresh metres. Both pred and GT trajectories are in the
        agent-local frame so the L2 error is directly interpretable in metres.
        """
        ades, fdes = [], []
        for i, result in enumerate(results):
            r = result.get('pts_bbox', result)
            info = self.data_infos[i]
            if 'gt_fut_traj' not in info or 'gt_velocity' not in info:
                continue

            mask = (info['valid_flag'] if self.use_valid_flag
                    else info['num_lidar_pts'] > 0)
            gt_boxes    = info['gt_boxes'][mask]        # [N, 9]
            gt_fut_traj = info['gt_fut_traj'][mask]     # [N, 6, 2]
            gt_velocity = info['gt_velocity'][mask]     # [N, 2]

            speed   = np.linalg.norm(gt_velocity, axis=-1)  # [N]
            moving  = speed > min_speed
            if not moving.any():
                continue

            gt_xy   = gt_boxes[moving, :2]   # [M, 2]
            gt_traj = gt_fut_traj[moving]    # [M, 6, 2]

            pred_xy   = r['motion_pred_xy']   # [K, 2]
            pred_traj = r['motion_preds']     # [K, 6, 2]

            # Greedy nearest-neighbour match in BEV
            dists     = np.linalg.norm(gt_xy[:, None] - pred_xy[None], axis=-1)  # [M, K]
            matched_k = dists.argmin(axis=1)                                       # [M]
            min_dists = dists[np.arange(len(gt_xy)), matched_k]                   # [M]
            valid     = min_dists < match_thresh

            for m, k in zip(np.where(valid)[0], matched_k[valid]):
                err = np.linalg.norm(pred_traj[k] - gt_traj[m], axis=-1)  # [6]
                ades.append(err.mean())
                fdes.append(err[-1])

        if not ades:
            return float('nan'), float('nan')
        return float(np.mean(ades)), float(np.mean(fdes))