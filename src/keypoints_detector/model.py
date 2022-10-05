from torchvision.models.detection import keypointrcnn_resnet50_fpn, rpn


def keypoint_detector(num_classes=4, num_keypoints=7, box_nms_thresh=0.2, rpn_score_thresh=0.80, box_score_thresh=0.80):
    anchor_generator = rpn.AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                           aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = keypointrcnn_resnet50_fpn(pretrained=False,
                                      pretrained_backbone=True,
                                      num_classes=num_classes,
                                      num_keypoints=num_keypoints,
                                      box_nms_thresh=box_nms_thresh,
                                      rpn_score_thresh=rpn_score_thresh,
                                      box_score_thresh=box_score_thresh,
                                      rpn_anchor_generator=anchor_generator)

    return model


if __name__ == '__main__':
    model = keypoint_detector()
    print(model)
