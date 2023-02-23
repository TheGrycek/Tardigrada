import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as func
from torch import nn
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection import rpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.roi_heads import fastrcnn_loss, keypointrcnn_inference, keypoints_to_heatmap
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.ops import nms
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

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


def resnet_fpn_extractor(backbone):
    returned_layers = [1, 2, 3, 4]
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [(backbone.inplanes // 8) * 2 ** (i - 1) for i in returned_layers]

    backbone = BackboneWithFPN(backbone,
                               return_layers,
                               in_channels_list,
                               out_channels=256,
                               extra_blocks=LastLevelMaxPool(),
                               norm_layer=None)
    return backbone


class KeypointRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for out_channels in layers:
            # d.append(nn.Dropout(0.2))
            d.append(nn.Conv2d(next_feature, out_channels, 3, stride=1, padding=1))
            # d.append(nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            d.append(nn.ReLU(inplace=True))
            next_feature = out_channels
        super().__init__(*d)
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


def keypoint_detector(num_classes=cfg.CLASSES_NUMBER,
                      num_keypoints=cfg.KEYPOINTS,
                      box_nms_thresh=cfg.BOX_NMS_THRESH,
                      rpn_score_thresh=cfg.RPN_SCORE_THRESH,
                      box_score_thresh=cfg.BOX_SCORE_THRESH,
                      box_detections_per_img=cfg.DETECTIONS_PER_IMG):

    anchor_generator = rpn.AnchorGenerator(sizes=(16, 32, 64, 128, 256),
                                           aspect_ratios=(0.2, 0.6, 1.0, 3))

    # used for keypoints_detector_old3.pth
    # anchor_generator = rpn.AnchorGenerator(sizes=(32, 64, 128, 256, 512),
    #                                        aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))

    weights_backbone = ResNet50_Weights.verify(ResNet50_Weights.IMAGENET1K_V2)
    backbone = resnet50(weights=weights_backbone, progress=True, norm_layer=nn.BatchNorm2d)
    backbone = resnet_fpn_extractor(backbone)

    keypoint_layers = tuple(512 for _ in range(8))
    keypoint_head = KeypointRCNNHeads(backbone.out_channels, keypoint_layers)

    model = KeypointRCNN(backbone,
                         keypoint_head=keypoint_head,
                         weights=None,
                         weights_backbone=ResNet50_Weights.DEFAULT,
                         num_classes=num_classes,
                         num_keypoints=num_keypoints,
                         box_nms_thresh=box_nms_thresh,
                         rpn_score_thresh=rpn_score_thresh,
                         box_score_thresh=box_score_thresh,
                         rpn_anchor_generator=anchor_generator,
                         box_detections_per_img=box_detections_per_img,
                         image_mean=[0.4950728416442871, 0.5257152915000916, 0.5137858986854553],
                         image_std=[0.08277688175439835, 0.0893404483795166, 0.08817232400178909]
                         )

    model.roi_heads.forward = types.MethodType(forward, model.roi_heads)

    return model


class KeypointDetector:
    def __init__(
            self,
            model_path=cfg.MODEL_PATH,
            rpn_score_thresh=cfg.RPN_SCORE_THRESH,
            box_score_thresh=cfg.BOX_SCORE_THRESH,
            box_nms_thresh=cfg.BOX_NMS_THRESH,
            tiling_nms_thresh=0.4,
            tile_grid_size=500,
            tiling_overlap=150,
            device=cfg.DEVICE,
            tiling=True
    ):
        self.model_path = model_path
        self.device = device
        self.rpn_score_thresh = rpn_score_thresh
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.tiling_nms_thresh = tiling_nms_thresh
        self.tile_grid_size = tile_grid_size
        self.tiling_overlap = tiling_overlap
        self.model = self.load_model()
        self.tiling = tiling

    def load_model(self):
        model = keypoint_detector(box_nms_thresh=self.box_nms_thresh,
                                  rpn_score_thresh=self.rpn_score_thresh,
                                  box_score_thresh=self.box_score_thresh)

        checkpoint = torch.load(self.model_path)["model"]
        model.load_state_dict(checkpoint)
        model.eval().to(self.device)

        return model

    def __call__(self, img):
        image_tensor = func.to_tensor(img).to(self.device)
        full_predicted = self.model([image_tensor])[0]
        full_predicted = {key: val.cpu().detach() for key, val in full_predicted.items()}

        if self.tiling:
            tile_predicted = self.tile_prediction(img)
            for key, val in tile_predicted.items():
                full_predicted[key] = torch.cat((tile_predicted[key], full_predicted[key]), dim=0)

            kpts_scores_sum = torch.sum(full_predicted["keypoints_scores"], dim=1)
            indices = nms(full_predicted["boxes"], kpts_scores_sum, self.tiling_nms_thresh)

            for key in full_predicted.keys():
                full_predicted[key] = full_predicted[key][indices]

        return full_predicted

    def tile_prediction(self, img):
        h, w, c = img.shape
        pad = self.tiling_overlap * 2
        tile_size = self.tile_grid_size + pad
        half_tile_size = tile_size / 2
        row_num = int(np.ceil(h / self.tile_grid_size))
        col_num = int(np.ceil(w / self.tile_grid_size))
        resize_ratios = (row_num * self.tile_grid_size / h, col_num * self.tile_grid_size / w)

        img_resized = cv2.resize(img, (col_num * self.tile_grid_size, row_num * self.tile_grid_size))

        start_pos = self.tiling_overlap + (self.tile_grid_size / 2)
        end_pos_diff = self.tiling_overlap + (self.tile_grid_size / 2)
        row_centers = np.linspace(start_pos, row_num * self.tile_grid_size - end_pos_diff, num=row_num)
        col_centers = np.linspace(start_pos, col_num * self.tile_grid_size - end_pos_diff, num=col_num)

        result_keys = ("boxes", "labels", "scores", "keypoints", "keypoints_scores")
        tile_predicted = {key: torch.Tensor([]) for key in result_keys}

        for row in range(row_num):
            for col in range(col_num):
                y_min = int(row_centers[row] - half_tile_size)
                y_max = int(row_centers[row] + half_tile_size)
                x_min = int(col_centers[col] - half_tile_size)
                x_max = int(col_centers[col] + half_tile_size)
                img_crop = img_resized[y_min: y_max, x_min: x_max, :]
                crop_tensor = func.to_tensor(img_crop).to(self.device)
                predicted = self.model([crop_tensor])[0]

                predicted["boxes"][:, [0, 2]] += x_min
                predicted["boxes"][:, [1, 3]] += y_min
                predicted["boxes"][:, [0, 2]] /= resize_ratios[1]
                predicted["boxes"][:, [1, 3]] /= resize_ratios[0]

                predicted["keypoints"][:, :, 0] += x_min
                predicted["keypoints"][:, :, 1] += y_min
                predicted["keypoints"][:, :, 0] /= resize_ratios[1]
                predicted["keypoints"][:, :, 1] /= resize_ratios[0]

                for key, val in tile_predicted.items():
                    tile_predicted[key] = torch.cat((val, predicted[key].cpu().detach()), dim=0)

        return tile_predicted


if __name__ == '__main__':
    model = keypoint_detector()
    print(model)
