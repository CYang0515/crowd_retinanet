import numpy as np
import torch
import torch.nn as nn
only_full = True # if true only detect full human else detect full and head human

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.75  # 0.25
        gamma = 2.0
        ignores = annotations[:, :, [-1]]
        annotations = annotations[:, :, 0: -1]
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]

            ignore = ignores[j, :, :]
            ignore = ignore[ignore[:, -1] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, 4: 8])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            assigned_ignores = ignore[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1

            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                assigned_ignores = assigned_ignores[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths_h = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights_h = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x_h = assigned_annotations[:, 0] + 0.5 * gt_widths_h
                gt_ctr_y_h = assigned_annotations[:, 1] + 0.5 * gt_heights_h

                gt_widths_f = assigned_annotations[:, 6] - assigned_annotations[:, 4]
                gt_heights_f = assigned_annotations[:, 7] - assigned_annotations[:, 5]
                gt_ctr_x_f = assigned_annotations[:, 4] + 0.5 * gt_widths_f
                gt_ctr_y_f = assigned_annotations[:, 5] + 0.5 * gt_heights_f

                # clip widths to 1
                gt_widths_h = torch.clamp(gt_widths_h, min=1)
                gt_heights_h = torch.clamp(gt_heights_h, min=1)

                gt_widths_f = torch.clamp(gt_widths_f, min=1)
                gt_heights_f = torch.clamp(gt_heights_f, min=1)

                targets_dx_f = (gt_ctr_x_f - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy_f = (gt_ctr_y_f - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw_f = torch.log(gt_widths_f / anchor_widths_pi)
                targets_dh_f = torch.log(gt_heights_f / anchor_heights_pi)

                targets_dx_h = (gt_ctr_x_h - anchor_ctr_x_pi) / anchor_widths_pi * 4
                targets_dy_h = (gt_ctr_y_h - anchor_ctr_y_pi) / anchor_heights_pi * 4
                targets_dw_h = torch.log(gt_widths_h / anchor_widths_pi * 4)
                targets_dh_h = torch.log(gt_heights_h / anchor_heights_pi * 4)

                targets = torch.stack((targets_dx_f, targets_dy_f, targets_dw_f, targets_dh_f,
                                       targets_dx_h, targets_dy_h, targets_dw_h, targets_dh_h))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]]).cuda()

                negative_indices = 1 - positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                weights = torch.ones(regression_diff.shape).cuda()
                if only_full:
                    weights[:, 4:] = 0
                else:
                    weights[:, 4:] = 1 - assigned_ignores
                regression_diff = regression_diff * weights

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                if only_full:
                    regression_losses.append(regression_loss[:, 0:4].mean())
                else:
                    regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0,
                                                                                                                 keepdim=True)


