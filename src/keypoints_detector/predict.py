import cv2
import keypoints_detector.config as cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from keypoints_detector.model import keypoint_detector


def predict(model, img, device):
    model.eval().to(device)

    image_tensor = F.to_tensor(img).to(device)
    predicted = model([image_tensor])[0]

    bboxes = predicted["boxes"].cpu().detach().numpy()
    labels = predicted["labels"].cpu().detach().numpy().astype(np.uint8)
    scores = predicted["scores"].cpu().detach().numpy()
    keypoints = predicted['keypoints'].cpu().detach().numpy()

    grouped_keypoints = np.round(keypoints, 0)[:, :, :2]

    for i in range(len(bboxes)):
        bbox, points = bboxes[i], grouped_keypoints[i]
        pt1 = tuple(bbox[:2].astype(np.uint16))
        pt2 = tuple(bbox[2:].astype(np.uint16))

        img = cv2.rectangle(img.copy(), pt1, pt2, (0, 0, 255), 2)

        position = (int(bboxes[i][0]), int(bboxes[i][1]))
        img = cv2.putText(img, f"{cfg.INSTANCE_CATEGORY_NAMES[labels[i]]}{i}",
                          position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 0, 0), thickness=2)

        img = cv2.line(img, tuple(points[-2]), tuple(points[-1]), color=(0, 255, 255), thickness=2)

        for ind in (0, 4):
            img = cv2.circle(img, center=tuple(points[ind].astype(np.int32)),
                             radius=4, color=(0, 255, 0), thickness=4)

        for j in range(len(keypoints[i])):
            img = cv2.circle(img, center=(round(keypoints[i][j][0]), round(keypoints[i][j][1])),
                             radius=4, color=(255, 0, 255), thickness=4)

    output = {"image": img,
              "keypoints": grouped_keypoints,
              "bboxes": bboxes,
              "labels": labels,
              "scores": scores}

    return output


def show_prediction(img):
    model = keypoint_detector()
    model.load_state_dict(torch.load("keypoints_detector/checkpoints/keypoints_detector_old.pth"))
    prediction = predict(model, img, cfg.DEVICE)
    cv2.imwrite("keypoint_rcnn_detection.jpg", prediction["image"])

    plt.figure(figsize=(20, 30))
    plt.imshow(prediction["image"])
    plt.xticks([])
    plt.yticks([])
    plt.show()
