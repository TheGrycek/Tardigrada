import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

import config as cfg


def keypoint_detector():
    torch.manual_seed(1)
    model = keypointrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True,
                                      num_classes=cfg.CLASSES_NUMBER, num_keypoints=cfg.KEYPOINTS)

    return model


if __name__ == '__main__':
    model = keypoint_detector()
    print(model)
