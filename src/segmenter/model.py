import cv2
import torch
from torch.nn import Conv2d, Linear
from torchvision.models.detection import maskrcnn_resnet50_fpn


def segmenter():
    model = maskrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    torch.manual_seed(1)
    new_conv2d = Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    model.roi_heads.mask_predictor.mask_fcn_logits = new_conv2d

    new_linear = Linear(in_features=1024, out_features=3, bias=True)
    model.roi_heads.box_predictor.cls_score = new_linear

    return model


def filter_cnts(cnts):
    cnts_filtered = []
    bboxes_fit = []
    bboxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 1000:
            continue
        rect_tilted = cv2.minAreaRect(cnt)
        if 0.5 < rect_tilted[1][0] / rect_tilted[1][1] < 2:
            continue
        rect = cv2.boundingRect(cnt)
        bboxes.append(rect)
        cnts_filtered.append(cnt)
        bboxes_fit.append(rect_tilted)

    return cnts_filtered, bboxes_fit, bboxes


def simple_segmenter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts, bboxes_fit, bboxes = filter_cnts(cnts)

    return cnts, bboxes_fit, bboxes, thresh
