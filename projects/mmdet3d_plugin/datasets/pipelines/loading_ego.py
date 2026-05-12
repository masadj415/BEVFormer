import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadFutureEgoWaypoints:
    """Convert gt_ego_waypoints (set by CustomNuScenesDataset.get_data_info)
    from numpy (N, 2) to a stacked DC tensor for batching.

    Only used in the train pipeline. Missing future frames are already
    padded with -999 by the dataset; the loss ignores those positions.
    """

    def __call__(self, results):
        wp = results.get('gt_ego_waypoints', None)
        if wp is None:
            return results
        wp_tensor = torch.from_numpy(np.asarray(wp, dtype=np.float32))  # (N, 2)
        results['gt_ego_waypoints'] = DC(wp_tensor, cpu_only=False, stack=True, pad_dims=None)        
        return results
