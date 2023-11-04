from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
from ultralytics.data.utils import autosplit

import keypoints_detector.config as cfg


def convert_labels():
    labels_dir = Path(cfg.ANNOTATION_FILE_PATH).parent
    convert_coco(labels_dir=str(labels_dir), use_keypoints=True)


def train():
    dataset_path = Path(cfg.IMAGES_PATH).parent
    if not Path(f"{dataset_path}/autosplit_train.txt").is_file():
        autosplit(path=dataset_path, weights=(0.9, 0.1, 0.0), annotated_only=False)

    save_dir = Path(f"{cfg.REPO_ROOT}/src/keypoints_detector/yolo/runs")
    model = YOLO('yolov8m-pose.yaml').load('yolov8m-pose.pt')

    results = model.train(
        data="./tardi-pose.yaml",
        cfg="params.yaml",
        project=save_dir
    )


if __name__ == "__main__":
    train()
