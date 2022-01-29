from torchvision.models.detection import keypointrcnn_resnet50_fpn


def keypoint_detector(num_classes=3, num_keypoints=4):
    model = keypointrcnn_resnet50_fpn(pretrained=False,
                                      pretrained_backbone=True,
                                      num_classes=num_classes,
                                      num_keypoints=num_keypoints,
                                      box_nms_thresh=0.4,
                                      rpn_score_thresh=0.90,
                                      box_score_thresh=0.90)

    return model


if __name__ == '__main__':
    model = keypoint_detector()
    print(model)
