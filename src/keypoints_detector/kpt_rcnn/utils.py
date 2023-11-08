import os
import random
from collections import OrderedDict

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
    ones = np.ones((keypoints.shape[0], 1))
    keypoints = np.concatenate((keypoints, ones), axis=1)
    keypoints = keypoints.reshape(-1, keypoints_num, 3)

    keypoints_tr, labels_tr = [], []
    for i in range(len(keypoints)):
        if i not in remove_ind:
            keypoints_tr.append(np.dot(M, keypoints[i].T).T)
            labels_tr.append(labels[int(i / keypoints_num)])

    keypoints_tr = np.round(np.array(keypoints_tr))

    return img, keypoints_tr[:, :, :2], np.array(bboxes_tr), labels_tr


def tensor2rgb(img):
    img = (img[0].numpy() * 255).astype(np.uint8)
    img = np.swapaxes(img, 0, 1)
    return np.swapaxes(img, 1, 2)


def calc_dist(pt1, pt2, tensor=False):
    if tensor:
        return torch.sqrt(torch.sum(torch.square(pt1 - pt2)))

    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def calc_dimensions(length_pts, width_pts, scale_ratio, tensor=False):
    if not tensor:
        width_pts = width_pts.astype(np.uint64)

    right, left = width_pts
    width = calc_dist(left, right, tensor=tensor) * scale_ratio
    len_parts = [calc_dist(length_pts[i], length_pts[i + 1], tensor=tensor) for i in range(len(length_pts) - 1)]

    if tensor:
        length = torch.sum(torch.tensor(len_parts)) * scale_ratio
    else:
        length = np.sum(np.asarray(len_parts)) * scale_ratio

    return length, width


def assign_prediction2ground_truth(pred_bboxes, targets_bboxes, iou_thresh=0.5):
    out = np.zeros((targets_bboxes.shape[0])) - 1
    ious = box_iou(pred_bboxes, targets_bboxes)

    for i in range(targets_bboxes.shape[0]):
        max_arg = torch.argmax(ious)
        row = (torch.div(max_arg, targets_bboxes.shape[0], rounding_mode='floor')).item()
        col = (max_arg - (row * targets_bboxes.shape[0])).item()

        if row in out:
            continue

        if ious[row, col].item() > iou_thresh:
            out[col] = int(row)

        ious[row, :] = 0
        ious[:, col] = 0

    return out


def get_assigned_data(prediction, target):
    idx = assign_prediction2ground_truth(prediction["boxes"], target["boxes"])

    # TODO: check correctness

    prediction_assigned = {
        "keypoints": prediction["keypoints"][:, :, :2][idx[idx >= 0], :, :],
        "boxes": prediction["boxes"][idx[idx >= 0], :],
        "scores": prediction["scores"][idx[idx >= 0]]
    }

    target_assigned = {
        "keypoints": target["keypoints"][:, :, :2][idx >= 0],
        "boxes": target["boxes"][idx >= 0]
    }
    return prediction_assigned, target_assigned


def calc_oks(predictions, targets, img_sizes):
    """Calculate object keypoint similarity"""
    oks_sum = 0
    total_pts = 0
    # TODO: define kappa parameters
    k = torch.tensor([1., 1., 1., 1., 1., 1., 1.])
    # k = torch.tensor([0.025, 0.072, 0.072, 0.072, 0.025, 0.062, 0.062])
    for prediction, target, img_size in zip(predictions, targets, img_sizes):
        if len(prediction["boxes"]) > 0:
            try:
                pred, targ = get_assigned_data(prediction, target)
                ws = torch.abs(targ["boxes"][:, 0] - targ["boxes"][:, 2])
                hs = torch.abs(targ["boxes"][:, 1] - targ["boxes"][:, 3])
                scales = (ws * hs) / np.prod(img_size)
                distances = (pred["keypoints"] - targ["keypoints"]).pow(2).sum(2).sqrt()
                kappa = torch.stack([k for _ in range(len(scales))])

                oks_sum += torch.exp(-(distances ** 2) / (2 * (scales.unsqueeze(1) ** 2) * (kappa ** 2))).sum()
                total_pts += target["keypoints"][:, :, 0].numel()
            except:
                pass

    return oks_sum / total_pts if total_pts > 0 else 0


def set_reproducibility_params():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def create_losses_dict():
    names = [
        "loss_total",
        "loss_classifier",
        "loss_box_reg",
        "loss_keypoint",
        "loss_objectness",
        "loss_rpn_box_reg"
    ]
    losses_names = [stage + name for stage in ["train_", "val_"] for name in names]
    losses = OrderedDict({key: [] for key in losses_names})

    return losses_names, losses


def tile_image(img, side=500, tile_padding=150):
    h, w, c = img.shape
    pad = (2 * tile_padding)
    tile_size = side + pad
    row_num, col_num = int(np.ceil(h / side)), int(np.ceil(w / side))
    resize_ratios = (row_num * tile_size / h, col_num * tile_size / w)

    img_resized = cv2.resize(img, (col_num * side, row_num * side))
    row_centers = np.linspace(tile_padding + (side / 2), row_num * side - tile_padding - (side / 2), num=row_num)
    col_centers = np.linspace(tile_padding + (side / 2), col_num * side - tile_padding - (side / 2), num=col_num)

    tiles = []
    for row in range(row_num):
        tiles_col = []
        for col in range(col_num):
            y_min = int(row_centers[row] - (tile_size / 2))
            y_max = int(row_centers[row] + (tile_size / 2))
            x_min = int(col_centers[col] - (tile_size / 2))
            x_max = int(col_centers[col] + (tile_size / 2))
            img_crop = img_resized[y_min: y_max, x_min: x_max, :]

            tiles_col.append(img_crop)
        tiles.append(tiles_col)

    return tiles, resize_ratios


def random_bbox_crop_roi(bboxes, img_shape, side_len=800):
    if img_shape[0] < side_len or img_shape[1] < side_len:
        return np.array([0, 0, img_shape[1], img_shape[0]])

    bbox = bboxes[random.randint(0, bboxes.shape[0] - 1)]
    half_len = (side_len // 2)

    x_center = np.mean([bbox[[0, 2]]])
    y_center = np.mean([bbox[[1, 3]]])

    crop = np.array([x_center - half_len,
                     y_center - half_len,
                     x_center + half_len,
                     y_center + half_len], dtype=int)

    d_x_left = min([0, crop[0]])
    d_x_right = min([0, img_shape[1] - crop[2]])
    d_y_top = min([0, crop[1]])
    d_y_bot = min([0, img_shape[0] - crop[3]])

    crop[[0, 2]] += abs(d_x_left) - abs(d_x_right)
    crop[[1, 3]] += abs(d_y_top) - abs(d_y_bot)

    return crop
