from .detrpose_hgnetv2_l import (
    model,
    postprocessor,
    public,
    private
)

public['hf_model'] = 'SebasJanampa/DETRPose_X_COCO'

model.backbone.name = 'B5'
model.encoder.dim_feedforward = 2048
model.transformer.reg_scale = 8

private.model.hidden_dim = 384