import numpy as np


def predict(model, img):
    result = model(img)[0]
    output = {
        "image": result.plot(),
        "keypoints": result.keypoints.xy.detach().cpu().numpy(),
        "bboxes": result.boxes.xyxy.detach().cpu().numpy(),
        "labels": result.boxes.cls.detach().cpu().numpy().astype(np.int8) + 1,  # YOLO doesn't have a background class
        "scores": result.boxes.conf.detach().cpu().numpy(),
        "kpt_scores": result.keypoints.conf.detach().cpu().numpy()
    }

    return output
