import re

from mmcv.runner.hooks import HOOKS
from mmcv.runner.dist_utils import master_only

try:
    from mmcv.runner.hooks.logger import WandbLoggerHook
except ImportError:
    from mmcv.runner.hooks.logger.wandb import WandbLoggerHook


@HOOKS.register_module()
class FilteredWandbLoggerHook(WandbLoggerHook):
    """
    Logs training metrics normally, but filters validation metrics so that
    only NDS and mAP are sent to Weights & Biases.

    Important: in distributed training, only rank 0 is allowed to log to W&B.
    """

    def __init__(self, keep_train=True, **kwargs):
        super().__init__(**kwargs)
        self.keep_train = keep_train

    def _is_validation_metric(self, key):
        return (
            "NuScenes" in key
            or key.startswith("pts_bbox_")
            or "label_aps" in key
            or "label_tp_errors" in key
            or "tp_errors" in key
        )

    def _keep_validation_metric(self, key):
        return (
            key.endswith("/NDS")
            or key.endswith("/mAP")
            or key == "NDS"
            or key == "mAP"
            or re.search(r"(^|/)NDS$", key) is not None
            or re.search(r"(^|/)mAP$", key) is not None
        )

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)

        filtered = {}
        for key, value in tags.items():
            if self._is_validation_metric(key):
                if self._keep_validation_metric(key):
                    filtered[key] = value
            else:
                if self.keep_train:
                    filtered[key] = value

        if len(filtered) == 0:
            return

        log_kwargs = {}
        if getattr(self, "with_step", True):
            log_kwargs["step"] = self.get_iter(runner)

        if hasattr(self, "commit"):
            log_kwargs["commit"] = self.commit

        self.wandb.log(filtered, **log_kwargs)
