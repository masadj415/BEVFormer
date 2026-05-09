from .nuscenes_dataset import CustomNuScenesDataset

try:
    from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2
    _has_v2 = True
except ImportError:
    _has_v2 = False

from .builder import custom_build_dataset
__all__ = ['CustomNuScenesDataset']
if _has_v2:
    __all__.append('CustomNuScenesDatasetV2')
