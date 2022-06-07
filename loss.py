import torch.nn as nn
import torch.nn.functional as F
import torch

from util import BezierSampler, HungarianMatcher, cubic_bezier_curve_segment


class HungarianBezierLoss(nn.Module):
    def __init__(self, weight=[0.4, 1], weight_seg=[0.4, 1], curve_weight=1, label_weight=0.1, seg_weight=0.75, alpha=0.8, sample_points=100, degree=3, k=9, reduction='mean', ignore_index=255):
        super().__init__()
        self.weight = torch.Tensor(weight)
        self.weight_seg = torch.Tensor(weight_seg)
        self.curve_weight = curve_weight
        self.label_weight = label_weight
        self.seg_weight = seg_weight
        self.alpha = alpha
        self.sample_points = sample_points
        self.degree = degree
        self.k = k
        self.reduction = reduction
        self.ignore_index = ignore_index

        self.pos_weight = torch.Tensor([weight[1] / weight[0]]).cuda()
        self.pos_weight_seg = torch.Tensor([weight_seg[1] / weight_seg[0]]).cuda()

        self.bezier_sampler = BezierSampler(sample_points)
        self.hungarian_matcher = HungarianMatcher(degree, sample_points, alpha, k)

    def forward(self, prediction, targets):
        # logits, curves, segmentations

        # Transform prediction and outputs
        output_curves = prediction[1]
        target_labels = torch.zeros_like(prediction[0])
        target_segmentations = torch.stack([target['segmentation_mask'] for target in targets])

        # Match the prediction and targets
        indices = self.hungarian_matcher.match(prediction[0], prediction[1], targets)
        idx = self.hungarian_matcher.get_src_permutation_idx(indices)
        output_curves = output_curves[idx]

        # Targets (rearrange each lane in the whole batch)
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_sample_points = torch.cat([t['sample_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Valid bezier segments
        target_keypoints = cubic_bezier_curve_segment(target_keypoints, target_sample_points)
        target_sample_points = self.bezier_sampler.sample_points(target_keypoints)

        target_labels[idx] = 1

        # Calculate loss
        classification_loss = self.compute_classification_loss(prediction[0], target_labels)
        curve_loss = self.compute_curve_loss(self.bezier_sampler.sample_points(output_curves), target_sample_points)
        segmentation_loss = self.compute_segmentation_loss(prediction[2], target_segmentations)

        total_loss = self.curve_weight * curve_loss + self.label_weight * classification_loss + self.seg_weight * segmentation_loss
        return total_loss, {'classification_loss': classification_loss, 'curve_loss': curve_loss, 'segmentation_loss': segmentation_loss}

    def compute_curve_loss(self, pred, target):
        
        if target.numel() == 0:
            target = pred.clone().detach()
        loss = F.l1_loss(pred, target, reduction='none')
        
        normalizer = target.shape[0] * target.shape[1]
        normalizer = torch.as_tensor([normalizer], dtype=pred.dtype, device=pred.device)

        loss = loss.sum() / normalizer
        
        return loss

    def compute_classification_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight, reduction=self.reduction) / self.pos_weight

    def compute_segmentation_loss(self, pred, target):
        # Process inputs
        pred = torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)

        # Process targets
        valid_map = (targets != self.ignore_index)
        targets[~valid_map] = 0
        targets = targets.float()

        return (F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight, reduction='none') / self.pos_weight * valid_map).mean()

        


