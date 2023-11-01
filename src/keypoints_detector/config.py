from pathlib import Path

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ROOT = "/tarmass" if Path("/.dockerenv").is_file() else "/home/grycek/Desktop/repos/Tardigrada"
ANNOTATION_FILE_PATH = f"/{REPO_ROOT}/src/keypoints_detector/datasets/train/dataset_537/TardigradaNew_537.json"
IMAGES_PATH = f"/{REPO_ROOT}/src/keypoints_detector/datasets/train/dataset_537"
MODEL_PATH = f"/{REPO_ROOT}/src/keypoints_detector/checkpoints/keypoints_detector_best.pth"

KEYPOINTS = 7
INSTANCE_CATEGORY_NAMES = ['__background__', 'eutar_bla', 'heter_ech', 'eutar_tra']
CLASSES_NUMBER = len(INSTANCE_CATEGORY_NAMES)
LOSS_WEIGHTS = {
    "loss_classifier": 1.0,
    "loss_box_reg": 1.0,
    "loss_keypoint": 1.0,
    "loss_objectness": 1.0,
    "loss_rpn_box_reg": 1.0
}
RANDOM_LOSS_WEIGHTS = False
CHECKPOINT_SAVE_INTERVAL = 20

# OPTIMIZER
LEARNING_RATE = 0.001
MOMENTUM = 0.9
DAMPENING = 0.0
WEIGHT_DECAY = 0.0001
NESTEROV = False
# SCHEDULER
GAMMA = 0.5
# MILESTONES = [700, 800]
MILESTONES = []
EPOCHS = 200
# DATALOADERS
BATCH_SIZE = 1
TEST_RATIO = 0.1
VAL_RATIO = 0.1
NUM_WORKERS = 10
TRANSFORM_TRAIN = True
SHUFFLE_TRAIN = True
# FREEZE
FREEZE_EPOCHS = []  # after these epochs selected layers will be frozen
UNFREEZE_EPOCHS = []  # after these epochs selected layers will be unfrozen
FREEZE_LAYERS = ["backbone", "rpn", "roi_heads.box_head", "roi_heads.box_predictor"]
# INFERENCE
RPN_SCORE_THRESH = 0.5
BOX_SCORE_THRESH = 0.7
BOX_NMS_THRESH = 0.5
DETECTIONS_PER_IMG = 300
