from .detrpose_hgnetv2_l import (
    model,
    postprocessor,
    public,
    private
)

public['hf_model'] = 'SebasJanampa/DETRPose_N_COCO'

model.backbone.name = 'B0'
model.backbone.use_lab = True
model.backbone.return_idx = [2, 3]

model.encoder.in_channels = [512, 1024]
model.encoder.use_encoder_idx = [1]
model.encoder.depth_mult = 0.5
model.encoder.expansion = 0.34
model.encoder.dim_feedforward = 512

model.transformer.num_layers = 3
model.transformer.dim_feedforward = 512
model.transformer.num_points= 6

private.model.hidden_dim = 128
private.model.num_levels = 2
private.model.feat_strides = [16, 32]
