import types

import torch
import torch.nn.functional as F
from torchvision.models.detection import keypointrcnn_resnet50_fpn, rpn
from torchvision.models.detection.roi_heads import fastrcnn_loss, keypointrcnn_inference, keypoints_to_heatmap
from torchvision.models.resnet import ResNet50_Weights

import keypoints_detector.config as cfg


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
    """Function from torchvision.models.detection.roi_heads"""
    N, K, H, W = keypoint_logits.shape
    if H != W:
        raise ValueError(
            f"keypoint_logits height and width (last two elements of shape) should be equal. "
            f"Instead got H = {H} and W = {W}"
        )
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.where(valid)[0]

    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    return keypoint_loss


def forward(self, features, proposals, image_shapes, targets=None):
    """
    RoIHeads method from torchvision.models.detection.roi_heads
    Args:
        features (List[Tensor])
        proposals (List[Tensor[N, 4]])
        image_shapes (List[Tuple[H, W]])
        targets (List[Dict])
    """
    if targets is not None:
        for t in targets:
            floating_point_types = (torch.float, torch.double, torch.half)
            if not t["boxes"].dtype in floating_point_types:
                raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
            if not t["labels"].dtype == torch.int64:
                raise TypeError("target labels must of int64 type, instead got {t['labels'].dtype}")
            if self.has_keypoint():
                if not t["keypoints"].dtype == torch.float32:
                    raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

    if self.training:
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    result = []
    losses = {}
    if self.training:
        if labels is None:
            raise ValueError("labels cannot be None")
        if regression_targets is None:
            raise ValueError("regression_targets cannot be None")
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    else:
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

    keypoint_proposals = [p["boxes"] for p in result]
    if self.training:
        num_images = len(proposals)
        keypoint_proposals = []
        pos_matched_idxs = []
        if matched_idxs is None:
            raise ValueError("if in trainning, matched_idxs should not be None")

        for img_id in range(num_images):
            pos = torch.where(labels[img_id] > 0)[0]
            keypoint_proposals.append(proposals[img_id][pos])
            pos_matched_idxs.append(matched_idxs[img_id][pos])
    else:
        pos_matched_idxs = None

    keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
    keypoint_features = self.keypoint_head(keypoint_features)
    keypoint_logits = self.keypoint_predictor(keypoint_features)

    loss_keypoint = {}
    if self.training:
        if targets is None or pos_matched_idxs is None:
            raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

        gt_keypoints = [t["keypoints"] for t in targets]
        rcnn_loss_keypoint = keypointrcnn_loss(
            keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
        )
        loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
    else:
        if keypoint_logits is None or keypoint_proposals is None:
            raise ValueError(
                "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
            )

        keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
        for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
            r["keypoints"] = keypoint_prob
            r["keypoints_scores"] = kps
    losses.update(loss_keypoint)

    return result, losses


def keypoint_detector(num_classes=cfg.CLASSES_NUMBER,
                      num_keypoints=cfg.KEYPOINTS,
                      box_nms_thresh=cfg.BOX_NMS_THRESH,
                      rpn_score_thresh=cfg.RPN_SCORE_THRESH,
                      box_score_thresh=cfg.BOX_SCORE_THRESH,
                      box_detections_per_img=cfg.DETECTIONS_PER_IMG):

    anchor_generator = rpn.AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                           aspect_ratios=(0.25, 0.5, 1.0, 2.0))

    # used for keypoints_detector_old3.pth
    # anchor_generator = rpn.AnchorGenerator(sizes=(32, 64, 128, 256, 512),
    #                                        aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))

    model = keypointrcnn_resnet50_fpn(weights=None,
                                      weights_backbone=ResNet50_Weights.DEFAULT,
                                      num_classes=num_classes,
                                      num_keypoints=num_keypoints,
                                      box_nms_thresh=box_nms_thresh,
                                      rpn_score_thresh=rpn_score_thresh,
                                      box_score_thresh=box_score_thresh,
                                      rpn_anchor_generator=anchor_generator,
                                      box_detections_per_img=box_detections_per_img
                                      )

    model.roi_heads.forward = types.MethodType(forward, model.roi_heads)

    return model


if __name__ == '__main__':
    model = keypoint_detector()
    print(model)
