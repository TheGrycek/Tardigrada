from pprint import pprint

import cv2
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import keypoints_detector.config as cfg
from keypoints_detector.dataset import create_dataloaders
from keypoints_detector.model import keypoint_detector
from keypoints_detector.utils import calc_oks


def predict(model, img):
    predicted = model(img)

    bboxes = predicted["boxes"].numpy()
    labels = predicted["labels"].numpy().astype(np.uint8)
    scores = predicted["scores"].numpy()
    keypoints = predicted['keypoints'].numpy()

    grouped_keypoints = np.round(keypoints, 0)[:, :, :2]

    for i, (label, bbox, points) in enumerate(zip(labels, bboxes, grouped_keypoints)):
        pt1 = tuple(bbox[:2].astype(np.uint16))
        pt2 = tuple(bbox[2:].astype(np.uint16))

        img = cv2.rectangle(img.copy(), pt1, pt2, (0, 0, 255), 2)

        position = (int(bbox[0]), int(bbox[1]))
        img = cv2.putText(img, f"{cfg.INSTANCE_CATEGORY_NAMES[label]}{i}",
                          position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 0, 0), thickness=2)

        for ind in (0, 4):
            img = cv2.circle(img, center=tuple(points[ind].astype(np.int32)),
                             radius=4, color=(0, 255, 0), thickness=4)

        for j in range(len(points)):
            img = cv2.circle(img, center=(round(points[j][0]), round(points[j][1])),
                             radius=4, color=(255, 0, 255), thickness=4)

        for k in range(4):
            img = cv2.line(img, tuple(points[k]), tuple(points[k + 1]), color=(0, 255, 255), thickness=2)

        img = cv2.line(img, tuple(points[-2]), tuple(points[-1]), color=(0, 255, 255), thickness=2)

    output = {"image": img,
              "keypoints": grouped_keypoints,
              "bboxes": bboxes,
              "labels": labels,
              "scores": scores}

    return output


# TODO: refactor testing, and move some code to benchmark.py
def test(model, dataloader_test, device=cfg.DEVICE):
    mAP = MeanAveragePrecision()
    predictions_, targets_, img_sizes = [], [], []

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader_test):
            for img, target in zip(images, targets):
                predicted = model([img.to(device)])[0]
                img_sizes.append(img.shape[:2])

                predictions_.append(dict(
                    boxes=predicted["boxes"].cpu().detach(),
                    scores=predicted["scores"].cpu().detach(),
                    labels=predicted["labels"].cpu().detach(),
                    keypoints=predicted["keypoints"].cpu().detach()
                ))

                targets_.append(dict(
                    boxes=target["boxes"].cpu().detach().type(torch.float),
                    labels=target["labels"].cpu().detach(),
                    keypoints=target["keypoints"].cpu().detach()
                ))

    mAP.update(predictions_, targets_)
    results = mAP.compute()
    results["oks"] = calc_oks(predictions_, targets_, img_sizes)

    return results


def run_testing(images_path=cfg.IMAGES_PATH, annotation_path=cfg.ANNOTATION_FILE_PATH,
                model_path=cfg.MODEL_PATH, device=cfg.DEVICE, model_config=None):
    model = keypoint_detector() if model_config is None else keypoint_detector(**model_config)
    model.load_state_dict(torch.load(str(model_path))["model"])
    model.eval().to(device)
    dataloaders = create_dataloaders(images_dir=str(images_path), annotation_file=str(annotation_path),
                                     val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO)

    return test(model, dataloaders["test"], device)


if __name__ == '__main__':
    print(f"MY DEVICE: {cfg.DEVICE}")
    results = run_testing()
    pprint(results)
