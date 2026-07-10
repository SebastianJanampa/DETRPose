from .detrpose_hgnetv2_l import (
    model,
    postprocessor,
    public,
    private
)

public['hf_model'] = 'SebasJanampa/DETRPose_S_COCO'

model.backbone.name = 'B0'
model.backbone.use_lab = True

model.encoder.in_channels = [256, 512, 1024]
model.encoder.depth_mult=0.34
model.encoder.expansion=0.5

model.decoder.num_layers = 3
