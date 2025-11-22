# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .aflw import AFLW
from .cofw import COFW
from .face300w import Face300W
from .wflw import WFLW

# === 1. 追加导入自定义数据集 ================================================
from .cleftlip import CleftLip, visualize_keypoints, calculate_accuracy
# === 2. 更新 __all__ 列表 =====================================================
__all__ = [
    'AFLW', 'COFW', 'Face300W', 'WFLW',
    'CleftLip', 'visualize_keypoints', 'calculate_accuracy',
    'get_dataset'
]


def get_dataset(config):
    """Return the dataset class according to config.DATASET.DATASET"""
    name = config.DATASET.DATASET.upper()

    if name == 'AFLW':
        return AFLW
    elif name == 'COFW':
        return COFW
    elif name in ('300W', 'FACE300W'):
        return Face300W
    elif name == 'WFLW':
        return WFLW
    elif name == 'CLEFT_LIP':
        return CleftLip               # ← 加这一分支
    else:
        raise NotImplementedError(f'Unknown dataset: {name}')
