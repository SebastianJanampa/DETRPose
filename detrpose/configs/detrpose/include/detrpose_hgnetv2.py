import detrpose.nn as nn
from detrpose.core import LazyCall as L

model = L(nn.DETR)(
    backbone=L(nn.HGNetv2)(
        name='B4',
        use_lab=False,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False,
        ),
	encoder=L(nn.DETRPoseHybridEncoder)(
        in_channels=[512, 1024, 2048],
        feat_strides="${private.model.feat_strides}",
        num_levels="${private.model.num_levels}",
        hidden_dim="${private.model.hidden_dim}",
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        expansion=1.0,
        depth_mult=1.0,
        act='silu',
        temperatureH=20,
        temperatureW=20,
        eval_spatial_size= "${private.model.eval_spatial_size}"
		),
	transformer=L(nn.DETRPoseTransformer)(
        hidden_dim="${private.model.hidden_dim}",
        dropout=0.0,
        num_heads=8,
        num_queries="${private.model.num_queries}",
        num_select="${private.model.num_select}",
        dim_feedforward=1024,
        num_layers=6,
        return_intermediate_dec=True,
        activation='relu',
        num_levels="${private.model.num_levels}",
        num_points=4,
        learnable_tgt_init=True,
        two_stage_type='standard',
        num_keypoints="${private.model.num_keypoints}",
        num_classes="${private.model.num_classes}",
        dec_pred_class_embed_share = False,
        dec_pred_pose_embed_share = False,
        two_stage_class_embed_share=False,
        two_stage_bbox_embed_share=False,
        cls_no_bias = False,
        # new parameters
        feat_strides="${private.model.feat_strides}",
        eval_spatial_size="${private.model.eval_spatial_size}",
        reg_max=32,
        reg_scale=4
        ),
    )

postprocessor = L(nn.PostProcessor)(
    num_select="${private.model.num_select}",
    num_keypoints="${private.model.num_keypoints}"
    )
