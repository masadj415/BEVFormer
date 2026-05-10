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

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        
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
            maps=info.get('maps', {}),
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

    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None,
                 result_names=['pts_bbox'], show=False, out_dir=None,
                 pipeline=None):
        """Evaluate with both detection and segmentation metrics."""
        # Run the parent bbox evaluation
        result_dict = super().evaluate(
            results, metric=metric, logger=logger,
            jsonfile_prefix=jsonfile_prefix, result_names=result_names,
            show=show, out_dir=out_dir, pipeline=pipeline)

        # Segmentation mIoU evaluation
        if results and isinstance(results[0], dict):
            first = results[0].get('pts_bbox', results[0])
            if 'seg_preds' in first:
                seg_result = self._evaluate_seg(results, logger=logger)
                result_dict.update(seg_result)

        return result_dict

    def _evaluate_seg(self, results, iou_thr=0.5, logger=None):
        """Compute per-class IoU for map segmentation."""
        map_classes = [
            'drivable_area', 'ped_crossing', 'walkway',
            'stop_line', 'carpark_area', 'divider',
        ]
        _SEG_LAYER_IDX = {
            'drivable_area': 0, 'road_segment': 1, 'road_block': 2, 'lane': 3,
            'ped_crossing': 4, 'walkway': 5, 'stop_line': 6,
            'carpark_area': 7, 'road_divider': 8, 'lane_divider': 9,
        }
        _CLASS_TO_LAYERS = {
            'drivable_area': ['drivable_area'],
            'ped_crossing':  ['ped_crossing'],
            'walkway':       ['walkway'],
            'stop_line':     ['stop_line'],
            'carpark_area':  ['carpark_area'],
            'divider':       ['road_divider', 'lane_divider'],
        }
        num_classes = len(map_classes)
        intersection = np.zeros(num_classes)
        union        = np.zeros(num_classes)

        for i, res in enumerate(results):
            pred_raw = res.get('pts_bbox', res).get('seg_preds', None)
            if pred_raw is None:
                continue
            pred = (pred_raw > iou_thr).astype(np.float32)  # (C, H, W)

            # Load GT from NPZ
            maps = self.data_infos[i].get('maps', {})
            npz_path = maps.get('map_mask', None) if maps else None
            if npz_path is None or not npz_path:
                continue
            try:
                raw = np.load(npz_path)['arr_0'].astype(np.float32)  # (10, H, W)
            except Exception:
                continue
            _, H, W = raw.shape

            gt = np.zeros((num_classes, H, W), dtype=np.float32)
            for cls_idx, cls_name in enumerate(map_classes):
                for layer in _CLASS_TO_LAYERS[cls_name]:
                    ch = _SEG_LAYER_IDX.get(layer)
                    if ch is not None:
                        gt[cls_idx] = np.logical_or(gt[cls_idx], raw[ch])

            # Resize pred to GT resolution if needed
            if pred.shape[1:] != gt.shape[1:]:
                import torch
                import torch.nn.functional as F
                pred_t = torch.from_numpy(pred).unsqueeze(0)  # (1, C, H, W)
                pred_t = F.interpolate(pred_t, size=gt.shape[1:], mode='nearest')
                pred = pred_t.squeeze(0).numpy()

            for c in range(num_classes):
                p = pred[c].astype(bool)
                g = gt[c].astype(bool)
                intersection[c] += (p & g).sum()
                union[c]        += (p | g).sum()

        iou_per_class = np.zeros(num_classes)
        for c in range(num_classes):
            if union[c] > 0:
                iou_per_class[c] = intersection[c] / union[c]

        miou = iou_per_class.mean()
        seg_result = {'seg/mIoU': float(miou)}
        for c, name in enumerate(map_classes):
            seg_result[f'seg/{name}_IoU'] = float(iou_per_class[c])

        if logger is not None:
            mmcv.print_log(f'\nMap Segmentation mIoU: {miou:.4f}', logger)
            for c, name in enumerate(map_classes):
                mmcv.print_log(
                    f'  {name:20s}: {iou_per_class[c]:.4f}', logger)
        else:
            print(f'\nMap Segmentation mIoU: {miou:.4f}')
            for c, name in enumerate(map_classes):
                print(f'  {name:20s}: {iou_per_class[c]:.4f}')

        return seg_result

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
