from itertools import product

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as func
from torch import nn
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection import rpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.ops import nms
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

import keypoints_detector.config as cfg


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
            d.append(nn.Dropout(0.2))
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
                      box_detections_per_img=cfg.DETECTIONS_PER_IMG,
                      dataset_mean=cfg.DATASET_MEAN,
                      dataset_std=cfg.DATASET_STD):

    anchor_generator = rpn.AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                           aspect_ratios=(0.5, 1.0, 2.0))

    weights, backbone_nn = ResNet50_Weights, resnet50
    weights_backbone = weights.verify(weights.IMAGENET1K_V2)
    backbone = backbone_nn(weights=weights_backbone, progress=True, norm_layer=nn.BatchNorm2d)
    backbone = resnet_fpn_extractor(backbone)

    keypoint_layers = tuple(512 for _ in range(8))
    keypoint_head = KeypointRCNNHeads(backbone.out_channels, keypoint_layers)

    model = KeypointRCNN(backbone,
                         keypoint_head=keypoint_head,
                         weights=None,
                         weights_backbone=weights.DEFAULT,
                         num_classes=num_classes,
                         num_keypoints=num_keypoints,
                         box_nms_thresh=box_nms_thresh,
                         rpn_score_thresh=rpn_score_thresh,
                         box_score_thresh=box_score_thresh,
                         rpn_anchor_generator=anchor_generator,
                         box_detections_per_img=box_detections_per_img,
                         image_mean=dataset_mean,
                         image_std=dataset_std)

    return model


class KeypointDetector:
    def __init__(
            self,
            model_path=cfg.RCNN_MODEL_PATH,
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
            # TODO: do it in parallel
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

        for row, col in product(range(row_num), range(col_num)):
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
