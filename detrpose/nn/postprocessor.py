"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/)
Copyright (c) 2023 RT-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from GroupPose (https://github.com/Michel-liu/GroupPose/)
Copyright (c) 2023 GroupPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from ED-Pose (https://github.com/IDEA-Research/ED-Pose/)
Copyright (c) 2023 IDEA. All Rights Reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def mod(a, b):
    out = a - a // b * b
    return out


class PostProcessor(nn.Module):
    """
    This module converts the model's raw outputs into final predictions.
    It supports TensorRT/ONNX export via `deploy=True` and conditionally
    processes boxes, masks, and keypoints based on their presence.
    """

    def __init__(self, num_select=100, num_keypoints=17):
        super().__init__()
        self.num_select = num_select
        self.num_keypoints = num_keypoints
        self.deploy_mode = False
        self.remap_mscoco_category = True

    @torch.no_grad()
    def forward(self, outputs, target_sizes=None):
        out_logits = outputs['pred_logits']
        _, _, num_classes = out_logits.shape

        # 1. Calculate Probabilities and keep Top-K scoring queries
        prob = out_logits.sigmoid()
        scores, index = torch.topk(prob.flatten(1), self.num_select, dim=-1)

        labels = mod(index, num_classes)
        index = index // num_classes

        # 2. Extract Bounding Boxes (if predicted)
        out_bbox = outputs.get('pred_boxes', None)
        boxes = None
        if out_bbox is not None:
            topk_boxes_expand = index.unsqueeze(-1).expand(-1, -1, 4)
            boxes = torch.gather(out_bbox, 1, topk_boxes_expand)
            boxes = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
            boxes *= target_sizes.repeat(1, 2).unsqueeze(1)

        # 3. Extract Masks (if predicted)
        out_masks = outputs.get('pred_masks', None)
        masks = None
        if out_masks is not None:
            _, _, H_feat, W_feat = out_masks.shape
            topk_masks_expand = index[:, :, None, None].expand(-1, -1, 1, H_feat, W_feat)
            masks = torch.gather(out_masks, 1, topk_masks_expand)
            mask_tmp = []
            for i in len(masks):
                masks_i = F.interpolate(masks, size=target_sizes, mode="bilinear", align_corners=False)
                w, h = target_sizes[i]
                mask_i = F.interpolate(
                    masks_i,
                    size=(int(h), int(w)),
                    align_corners=False,
                )  # [K,1,H,W]
                mask_i = mask_i > 0.0
                mask_tmp.append(mask_i)
            masks = mask_tmp

        # 4. Extract Keypoints (if predicted)
        out_kpts = outputs.get('pred_keypoints', None)
        keypoints = None
        if out_kpts is not None:
            if out_kpts.dim() == 3:
                b, nq, total_kpt = out_kpts.shape
                kpts_dim = total_kpt // self.num_keypoints
                topk_kpts_expand = index.unsqueeze(-1).expand(-1, -1, total_kpt)
                keypoints = torch.gather(out_kpts, 1, topk_kpts_expand)
                keypoints = keypoints.unflatten(-1, (self.num_keypoints, kpts_dim))
                keypoints = keypoints[:, :, :, :2]
            elif out_kpts.dim() == 4:
                _, _, num_kpts, kpts_dim = out_kpts.shape
                topk_kpts_expand = index[..., None, None].expand(-1, -1, self.num_keypoints, 2)
                keypoints = torch.gather(out_kpts, 1,topk_kpts_expand)
            keypoints[:, :, :, :2] *= target_sizes[:, None, None, :]


        # ==========================================================
        # DEPLOYMENT MODE (TensorRT / ONNX)
        # Bypasses dicts/scaling and returns graph-safe raw tensors
        # ==========================================================
        if self.deploy_mode:
            # Strict requested order: scores, labels, boxes, masks, keypoints
            deploy_outputs = [scores, labels]
            if boxes is not None:
                deploy_outputs.append(boxes)
            if masks is not None:
                deploy_outputs.append(masks)
            if keypoints is not None:
                deploy_outputs.append(keypoints)

            return tuple(deploy_outputs)

        # ==========================================================
        # STANDARD PYTHON MODE (Training / Evaluation)
        # ==========================================================
        # if self.remap_mscoco_category:
        #     from src.utils.data.coco import mscoco_label2category
        #     labels = (
        #         torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
        #         .to(labels.device)
        #         .reshape(labels.shape)
        #     )

        assert target_sizes is not None, "target_sizes must be provided in non-deploy mode"
        if keypoints is not None:
            ones = torch.ones_like(keypoints[..., 0:1])
            keypoints = torch.cat([keypoints, ones], dim=-1).flatten(-2)

        results = []
        for i in range(len(scores)):
            # Strict insertion order: scores, labels, boxes, masks, keypoints
            res = {
                "scores": scores[i],
                "labels": labels[i],
            }

            if boxes is not None:
                res["boxes"] = boxes[i]

            if masks is not None:
                res["masks"] = masks[i]

            if keypoints is not None:
                res["keypoints"] = keypoints[i]

            results.append(res)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self