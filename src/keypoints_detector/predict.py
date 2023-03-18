import cv2
import numpy as np

import keypoints_detector.config as cfg


def predict(model, img):
    predicted = model(img)

    bboxes = predicted["boxes"].numpy()
    labels = predicted["labels"].numpy().astype(np.uint8)
    scores = predicted["scores"].numpy()
    keypoints = predicted['keypoints'].numpy()
    keypoint_scores = predicted["keypoints_scores"].numpy()

    grouped_keypoints = np.round(keypoints, 0)[:, :, :2]

    for i, (label, bbox, points) in enumerate(zip(labels, bboxes, grouped_keypoints)):
        pt1 = tuple(bbox[:2].astype(np.uint16))
        pt2 = tuple(bbox[2:].astype(np.uint16))
        position = (int(bbox[0]), int(bbox[1]))

        img = cv2.rectangle(img.copy(), pt1, pt2, (0, 0, 255), 2)
        img = cv2.putText(img, f"{cfg.INSTANCE_CATEGORY_NAMES[label]}{i}",
                          position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 0, 0), thickness=2)

        for j in (0, 4):
            img = cv2.circle(img, center=tuple(points[j].astype(np.int32)),
                             radius=4, color=(0, 255, 0), thickness=4)

        for pt_x, pt_y in points:
            img = cv2.circle(img, center=(round(pt_x), round(pt_y)), radius=4, color=(255, 0, 255), thickness=4)

        for k in range(4):
            img = cv2.line(img, tuple(points[k]), tuple(points[k + 1]), color=(0, 255, 255), thickness=2)

        img = cv2.line(img, tuple(points[-2]), tuple(points[-1]), color=(0, 255, 255), thickness=2)

    output = {"image": img,
              "keypoints": grouped_keypoints,
              "bboxes": bboxes,
              "labels": labels,
              "scores": scores,
              "kpt_scores": keypoint_scores}

    return output
