from .detrpose_hgnetv2_l import (
    model,
    postprocessor,
    private,
    public
)

public['hf_model'] = 'SebasJanampa/DETRPose_L_CROWDPOSE'

public.num_keypoints = 14
public.dataset_type = 'CROWDPOSE'
public.iou_types = ['keypoints_crowd']
