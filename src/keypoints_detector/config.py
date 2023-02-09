import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ROOT = os.environ["REPO_ROOT"] if "REPO_ROOT" in os.environ else "tarmass"
ANNOTATION_FILE_PATH = f"/{REPO_ROOT}/src/images/train/dataset_100/TardigradaNew.json"
IMAGES_PATH = f"/{REPO_ROOT}/src/images/train/dataset_100"
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
CHECKPOINT_SAVE_INTERVAL = 20

# OPTIMIZER
LEARNING_RATE = 0.0001
MOMENTUM = 0.1
DAMPENING = 0.0
WEIGHT_DECAY = 0.0
NESTEROV = False
# SCHEDULER
GAMMA = 0.5
MILESTONES = [700, 800]
EPOCHS = 1000
# DATALOADERS
BATCH_SIZE = 1
TEST_RATIO = 0.1
VAL_RATIO = 0.1
NUM_WORKERS = 10
# FREEZE
FREEZE_EPOCHS = []  # after these epochs selected layers will be frozen
UNFREEZE_EPOCHS = []  # after these epochs selected layers will be unfrozen
FREEZE_LAYERS = ["backbone", "rpn", "roi_heads.box_head", "roi_heads.box_predictor"]
# INFERENCE
RPN_SCORE_THRESH = 0.5
BOX_SCORE_THRESH = 0.7
BOX_NMS_THRESH = 0.5
DETECTIONS_PER_IMG = 300
