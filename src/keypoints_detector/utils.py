import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

seed = 123
np.random.seed(seed)


def scale_rotate(img, keypoints, bboxes, labels, max_angle=360, scale_range=(0.7, 1.0), keypoints_num=7):
    bboxes = np.array(bboxes)
    keypoints = np.array(keypoints)
    angle = np.random.uniform(0, max_angle)
    scale = np.random.uniform(low=scale_range[0], high=scale_range[1])

    # get rotation matrix
    h, w = img.shape[:2]
    c_x, c_y = w // 2, h // 2
    M = cv2.getRotationMatrix2D((c_x, c_y), angle, scale)

    # rotate image
    img = cv2.warpAffine(img, M, (w, h))
    h_rot, w_rot = img.shape[:2]

    # rotate bboxes
    # add the other corner points
    xs = bboxes[:, 0::2]
    xs_flip = np.flip(xs, axis=1)
    ys = bboxes[:, 1::2]
    ones = np.ones((xs.shape[0], 4))
    bboxes = np.concatenate((xs, xs_flip, ys, ys, ones), axis=1).reshape((-1, 3, 4))
    bboxes = np.round(np.array([np.dot(M, box).T for box in bboxes])).astype(int)

    # calculate new bbox shape
    bboxes_tr, remove_ind = [], []
    for i, box in enumerate(bboxes):
        x1 = np.clip(np.min(box[:, 0]), 0, w_rot)
        y1 = np.clip(np.min(box[:, 1]), 0, h_rot)
        x2 = np.clip(np.max(box[:, 0]), 0, w_rot)
        y2 = np.clip(np.max(box[:, 1]), 0, h_rot)

        if abs(x1 - x2) == 0 or abs(y1 - y2) == 0:
            remove_ind.append(i)
            continue

        bboxes_tr.append(np.array([x1, y1, x2, y2]))

    # rotate keypoints, remove redundant labels
    keypoints[:, :, 2] = 1
    keypoints_tr, labels_tr = [], []
    for i in range(len(keypoints)):
        if i not in remove_ind:
            keypoints_tr.append(np.dot(M, keypoints[i].T).T)
            labels_tr.append(labels[i])

    keypoints_tr = np.round(np.array(keypoints_tr))
    visible = np.ones((keypoints_tr.shape[0], keypoints_num, 1)) * 2
    keypoints_tr = np.dstack((keypoints_tr, visible))

    return img, keypoints_tr, np.array(bboxes_tr), labels_tr


def tensor2rgb(img):
    img = (img[0].numpy() * 255).astype(np.uint8)
    img = np.swapaxes(img, 0, 1)
    return np.swapaxes(img, 1, 2)


def calc_dist(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def assign_prediction2ground_truth(pred_bboxes, targets_bboxes, iou_thresh=0.5):
    ious = box_iou(targets_bboxes, pred_bboxes)
    max_args = torch.argmax(ious, dim=1)

    for i in range(len(max_args)):
        if ious[i][max_args[i]] < iou_thresh:
            max_args[i] = -1

    return max_args


def get_assigned_data(prediction, target):
    idx = assign_prediction2ground_truth(prediction["boxes"], target["boxes"])
    prediction_assigned = {
        "keypoints": prediction["keypoints"][:, :, :2][idx[idx >= 0], :, :],
        "boxes": prediction["boxes"][idx[idx >= 0], :],
        "scores": prediction["scores"][idx[idx >= 0]]
    }

    target_assigned = {
        "keypoints": target["keypoints"][:, :, :2][idx[idx >= 0], :, :],
        "boxes": target["boxes"][idx[idx >= 0], :]
    }
    return prediction_assigned, target_assigned


def calc_oks(predictions, targets, img_sizes):
    """Calculate object keypoint similarity"""
    oks_sum = 0
    total_pts = 0
    # TODO: define kappa parameters
    k = torch.tensor([0.025, 0.072, 0.072, 0.072, 0.025, 0.062, 0.062])
    for prediction, target, img_size in zip(predictions, targets, img_sizes):
        pred, targ = get_assigned_data(prediction, target)
        ws = torch.abs(targ["boxes"][:, 0] - targ["boxes"][:, 2])
        hs = torch.abs(targ["boxes"][:, 1] - targ["boxes"][:, 3])
        scales = ws * hs / np.sum(img_size)
        distances = (pred["keypoints"] - targ["keypoints"]).pow(2).sum(2).sqrt()
        kappa = torch.stack([k for _ in range(len(scales))])

        oks_sum += torch.exp(-(distances ** 2) / (2 * (scales.unsqueeze(1) ** 2) * (kappa ** 2))).sum()
        total_pts += target["keypoints"][:, :, 0].numel()

    return oks_sum / total_pts
