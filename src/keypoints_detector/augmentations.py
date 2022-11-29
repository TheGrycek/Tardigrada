import numpy as np
import cv2

seed = 123
np.random.seed(seed)


def scale_rotate(img, bboxes, keypoints, labels):
    bboxes = np.array(bboxes)
    keypoints = np.array(keypoints)
    angle = np.random.uniform(0, 180)
    scale = np.random.uniform(low=0.7, high=1.0)

    # rotate image
    h, w = img.shape[:2]
    c_x, c_y = w // 2, h // 2
    M = cv2.getRotationMatrix2D((c_x, c_y), angle, scale)
    img = cv2.warpAffine(img, M, (w, h))
    h_rot, w_rot = img.shape[:2]

    # rotate bboxes
    xs = bboxes[:, 0::2]
    xs_flip = np.flip(xs, axis=1)
    ys = bboxes[:, 1::2]
    ones = np.ones((xs.shape[0], 4))
    bboxes = np.concatenate((xs, xs_flip, ys, ys, ones), axis=1).reshape((-1, 3, 4))
    bboxes = np.round(np.array([np.dot(M, box).T for box in bboxes])).astype(int)

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
    num_keypoints = 7  # pass self.points_num = KEYPOINTS from Dataset class
    ones = np.ones((keypoints_tr.shape[0], num_keypoints, 1)) * 2
    keypoints_tr = np.dstack((keypoints_tr, ones))

    return img, np.array(bboxes_tr), keypoints_tr, labels_tr
