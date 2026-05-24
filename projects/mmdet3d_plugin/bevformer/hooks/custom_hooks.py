from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):

    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class AuxLossWarmupHook(Hook):
    """
    Linearly ramp ego-trajectory and motion-head loss weights from
    `min_weight` to their target values over [start_epoch, end_epoch].

    This prevents auxiliary gradients from conflicting with detection
    gradients before the shared BEV/query representations have stabilised.

    Before start_epoch  → weights held at min_weight
    start_epoch..end_epoch → linear ramp to target
    After end_epoch     → weights held at target

    Use in config:
        custom_hooks = [
            dict(type='AuxLossWarmupHook',
                 start_epoch=9, end_epoch=15,
                 ego_target=0.5, motion_target=0.25,
                 min_weight=0.02)
        ]
    """

    def __init__(self, start_epoch, end_epoch,
                 ego_target=0.5, motion_target=0.25, min_weight=0.02):
        assert end_epoch > start_epoch
        self.start_epoch  = start_epoch
        self.end_epoch    = end_epoch
        self.ego_target   = ego_target
        self.motion_target = motion_target
        self.min_weight   = min_weight

    def _scale(self, epoch):
        if epoch <= self.start_epoch:
            return 0.0
        if epoch >= self.end_epoch:
            return 1.0
        return (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

    def before_train_iter(self, runner):
        # runner.epoch is 0-indexed and counts completed epochs.
        # Use iter-level precision so the ramp is smooth within epochs.
        iters_per_epoch = len(runner.data_loader)
        elapsed = runner.epoch * iters_per_epoch + runner.inner_iter
        total_ramp = (self.end_epoch - self.start_epoch) * iters_per_epoch
        start_iter = self.start_epoch * iters_per_epoch

        if elapsed <= start_iter:
            scale = 0.0
        elif elapsed >= start_iter + total_ramp:
            scale = 1.0
        else:
            scale = (elapsed - start_iter) / total_ramp

        ego_w    = self.min_weight + scale * (self.ego_target    - self.min_weight)
        motion_w = self.min_weight + scale * (self.motion_target - self.min_weight)

        model = runner.model
        if hasattr(model, 'module'):   # unwrap DDP
            model = model.module

        if getattr(model, 'ego_trajectory_head', None) is not None:
            model.ego_trajectory_head.loss_weight = ego_w
        if getattr(model, 'motion_head', None) is not None:
            model.motion_head.loss_weight = motion_w

    def after_train_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        ego_w    = getattr(getattr(model, 'ego_trajectory_head', None), 'loss_weight', None)
        motion_w = getattr(getattr(model, 'motion_head', None),         'loss_weight', None)
        runner.logger.info(
            f'AuxLossWarmup — epoch {runner.epoch}  '
            f'ego_w={ego_w:.4f}  motion_w={motion_w:.4f}'
        )
