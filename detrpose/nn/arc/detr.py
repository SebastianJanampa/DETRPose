"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM (https://github.com/Intellindust-AI-Lab/DEIM/)
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/)
Copyright (c) 2023 RT-DETR Authors. All Rights Reserved.
"""

from torch import nn

class DETR(nn.Module):
    def __init__(
        self, 
        backbone, 
        encoder, 
        transformer
        ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.transformer = transformer

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    def forward(self, samples, targets=None):
        feats = self.backbone(samples)
        feats = self.encoder(feats)
        out = self.transformer(
            feats,
            targets,
            img_dim=samples.shape[-2:] if self.training else None
        )
        return out

