from torchvision.models.detection import keypointrcnn_resnet50_fpn, rpn


def keypoint_detector(num_classes=4, num_keypoints=7, box_nms_thresh=0.50, rpn_score_thresh=0.80, rpn_nms_thresh=0.70,
                      box_score_thresh=0.80, box_detections_per_img=300):
    anchor_generator = rpn.AnchorGenerator(sizes=(16, 32, 64, 128, 256),
                                           aspect_ratios=(0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0))

    model = keypointrcnn_resnet50_fpn(pretrained=False,
                                      pretrained_backbone=True,
                                      num_classes=num_classes,
                                      num_keypoints=num_keypoints,
                                      box_nms_thresh=box_nms_thresh,
                                      rpn_score_thresh=rpn_score_thresh,
                                      rpn_nms_thresh=rpn_nms_thresh,
                                      box_score_thresh=box_score_thresh,
                                      rpn_anchor_generator=anchor_generator,
                                      box_detections_per_img=box_detections_per_img,
                                      image_mean=[0.466063529253006, 0.5127472281455994, 0.490399032831192],
                                      image_std=[0.08915058523416519, 0.09907367825508118, 0.096004918217659]
                                      )

    return model


if __name__ == '__main__':
    model = keypoint_detector()
    print(model)
