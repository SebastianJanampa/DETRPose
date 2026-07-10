from .include.detrpose_hgnetv2 import model, postprocessor
from .include.common import public, private

__all__ = [
    'model',
    'postprocessor',
    'public',
    'private'
]

public['hf_model'] = 'SebasJanampa/DETRPose_L_COCO'
