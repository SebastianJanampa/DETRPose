from .detrpose_hgnetv2_l_crowdpose import (
    model,
    postprocessor,
    public,
    private
)

public['hf_model'] = 'SebasJanampa/DETRPose_M_CROWDPOSE'

model.backbone.name = 'B2'
model.backbone.use_lab = True
model.encoder.in_channels = [384, 768, 1536]
model.encoder.depth_mult = 0.67
model.decoder.num_layers = 4


