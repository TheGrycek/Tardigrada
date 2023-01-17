import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = os.environ["REPO_ROOT"] if "REPO_ROOT" in os.environ else "tarmass"
ANNOTATION_FILE_PATH = f"/{REPO_ROOT}/src/images/train/dataset_100/TardigradaNew.json"
IMAGES_PATH = f"/{REPO_ROOT}/src/images/train/dataset_100"
MODEL_PATH = f"/{REPO_ROOT}/src/keypoints_detector/checkpoints/keypoints_detector_old3.pth"

# TRAINING HYPER-PARAMETERS
LEARNING_RATE = 0.00017593
MOMENTUM = 0.39937871
WEIGHT_DECAY = 0.00003578
GAMMA = 0.5
MILESTONES = [200, 300, 600, 1000]
EPOCHS = 16000
BATCH_SIZE = 1
TEST_RATIO = 0.1
VAL_RATIO = 0.1
CHECKPOINT_SAVE_INTERVAL = 100
NUM_WORKERS = 10
INSTANCE_CATEGORY_NAMES = ['__background__', 'eutar_bla', 'heter_ech', 'eutar_tra']
CLASSES_NUMBER = len(INSTANCE_CATEGORY_NAMES)
KEYPOINTS = 7
LOSS_WEIGHTS = {
    "loss_classifier": 1.0,
    "loss_box_reg": 1.0,
    "loss_keypoint": 1.0,
    "loss_objectness": 1.0,
    "loss_rpn_box_reg": 1.0
}

# INFERENCE HYPER-PARAMETERS
RPN_SCORE_THRESH = 0.5
BOX_SCORE_THRESH = 0.7
BOX_NMS_THRESH = 0.5
DETECTIONS_PER_IMG = 300
