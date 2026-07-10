import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import sigmoid_focal_loss

def _to_one_hot(src_logits, tgt_classes, tgt_indices, num_classes):
    target_classes = torch.full(
        src_logits.shape[:2], num_classes,
        dtype=torch.int64, device=src_logits.device
    )
    target_classes[tgt_indices] = tgt_classes
    target = F.one_hot(target_classes, num_classes=num_classes + 1)[..., :-1]

    return target


def loss_labels_focal_func(
        src_logits,
        tgt_classes,
        tgt_indices,
        num_classes,
        focal_alpha=0.25,
        gamma=2,
    ):
    target = _to_one_hot(src_logits, tgt_classes, tgt_indices, num_classes)

    loss = sigmoid_focal_loss(
        src_logits, target,
        alpha=focal_alpha, gamma=gamma, reduction='none',
    )

    return loss.mean(1).sum()


def loss_labels_vfl_func(
        src_logits,
        tgt_classes,
        tgt_indices,
        num_classes,
        ious,
        focal_alpha=0.25,
        gamma=2,
    ):
    target = _to_one_hot(src_logits, tgt_classes, tgt_indices, num_classes)

    target_score_o = torch.zeros_like(tgt_classes, dtype=src_logits.dtype)
    target_score_o[tgt_indices] = ious.to(target_score_o.dtype)
    target_score = target_score_o.unsqueeze(-1) * target

    pred_score = F.sigmoid(src_logits).detach()
    weight = focal_alpha * pred_score.pow(gamma) * (1 - target) + target_score

    loss = F.binary_cross_entropy_with_logits(
        src_logits, target_score, weight=weight, reduction='none'
    )
    loss = loss.mean(1).sum()

    return loss


def loss_labels_mal_func(
        src_logits,
        tgt_classes,
        tgt_indices,
        num_classes,
        ious,
        mal_alpha=0.25,
        gamma=2,
    ):
    target = _to_one_hot(src_logits, tgt_classes, tgt_indices, num_classes)

    target_score_o = torch.zeros_like(tgt_classes, dtype=src_logits.dtype)
    target_score_o[tgt_indices] = ious.to(target_score_o.dtype)
    target_score = target_score_o.unsqueeze(-1) * target

    pred_score = F.sigmoid(src_logits).detach()
    target_score = target_score.pow(gamma)
    if mal_alpha != None:
        weight = mal_alpha * pred_score.pow(gamma) * (1 - target) + target
    else:
        weight = pred_score.pow(gamma) * (1 - target) + target

    loss = F.binary_cross_entropy_with_logits(
        src_logits, target_score, weight=weight, reduction='none'
    )
    loss = loss.mean(1).sum()

    return loss


def loss_local(self, outputs, targets, indices, num_boxes, T=5):
    """Compute Fine-Grained Localization (FGL) Loss
    and Decoupled Distillation Focal (DDF) Loss."""

    losses = {}
    if "pred_corners" in outputs and "teacher_corners" in outputs:
        idx = self._get_src_permutation_idx(indices)
        reg_max = outputs['reg_max']

        # compute oks
        src_keypoints = outputs['pred_keypoints'][idx]  # xyxyvv

        Z_pred = src_keypoints[:, 0:(self.num_body_points * 2)]
        targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        targets_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        Z_gt = targets_keypoints[:, 0:(self.num_body_points * 2)]
        V_gt = targets_keypoints[:, (self.num_body_points * 2):]

        oks = self.oks(
            Z_pred, Z_gt, V_gt, targets_area,
            weight=None, avg_factor=None, reduction_override=None
        )

        # The ddf los starts here
        pred_corners = outputs["pred_corners"].reshape(-1, (reg_max + 1))
        target_corners = outputs["teacher_corners"].reshape(-1, (reg_max + 1))
        if torch.equal(pred_corners, target_corners):
            losses["loss_dfl"] = pred_corners.sum() * 0
        else:
            weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

            mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
            mask[idx] = True
            mask = mask.unsqueeze(-1).repeat(1, 1, (self.num_body_points * 2)).reshape(-1)
            weight_targets_local[idx] = oks.reshape_as(weight_targets_local[idx]).to(
                weight_targets_local.dtype
            )
            weight_targets_local = (
                weight_targets_local.unsqueeze(-1).repeat(1, 1, (self.num_body_points * 2)).reshape(-1).detach()
            )
            loss_match_local = (
                    weight_targets_local
                    * (T ** 2)
                    * (
                        nn.KLDivLoss(reduction="none")(
                            F.log_softmax(pred_corners / T, dim=1),
                            F.softmax(target_corners.detach() / T, dim=1),
                        )
                    ).sum(-1)
            )
            if "is_dn" not in outputs:
                batch_scale = (
                        8 / outputs["pred_keypoints"].shape[0]
                )  # Avoid the influence of batch size per GPU
                self.num_pos, self.num_neg = (
                    (mask.sum() * batch_scale) ** 0.5,
                    ((~mask).sum() * batch_scale) ** 0.5,
                )
            loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
            loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
            losses["loss_dfl"] = (
                                         loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                                 ) / (self.num_pos + self.num_neg)

    return losses


def loss_keypoints(self, outputs, targets, indices, num_boxes):
    idx = self._get_src_permutation_idx(indices)
    src_keypoints = outputs['pred_keypoints'][idx]
    if len(src_keypoints) == 0:
        device = outputs["pred_logits"].device
        losses = {
            'loss_keypoints': torch.as_tensor(0., device=device) + src_keypoints.sum() * 0,
            'loss_oks': torch.as_tensor(0., device=device) + src_keypoints.sum() * 0,
        }
        return losses

    src_keypoints = src_keypoints[:, 0:(self.num_body_points * 2)]

    targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    tgt_keypoints = targets_keypoints[:, 0:(self.num_body_points * 2)]
    tgt_viz = targets_keypoints[:, (self.num_body_points * 2):]
    tgt_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    oks_loss = 1 - self.oks(src_keypoints, tgt_keypoints, tgt_viz, tgt_area, weight=None, avg_factor=None, reduction_override=None)

    pose_loss = F.l1_loss(src_keypoints, tgt_keypoints, reduction='none')
    pose_loss = pose_loss * tgt_viz.repeat_interleave(2, dim=1)

    losses = {}
    losses['loss_keypoints'] = pose_loss.sum() / num_boxes
    losses['loss_oks'] = oks_loss.sum() / num_boxes

    return losses