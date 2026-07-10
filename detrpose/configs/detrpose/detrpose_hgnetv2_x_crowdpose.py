from .detrpose_hgnetv2_l_crowdpose import (
    model,
    postprocessor,
    public,
    private
)

public['hf_model'] = 'SebasJanampa/DETRPose_X_CROWDPOSE'

model.backbone.name = 'B5'
model.encoder.dim_feedforward = 2048
model.decoder.reg_scale = 8

private.model.hidden_dim = 384
