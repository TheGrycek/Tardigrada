import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRAINING HYPER-PARAMETERS
LEARNING_RATE = 0.002 # 0.001
MOMENTUM = 0.1
WEIGHT_DECAY = 0.0001
GAMMA = 0.5
MILESTONES = [200, 400]
EPOCHS = 3000
# EPOCHS = 25
BATCH_SIZE = 1
TEST_RATIO = 0
VAL_RATIO = 0.99  # 0.1
CHECKPOINT_SAVE_INTERVAL = 20
NUM_WORKERS = 5
INSTANCE_CATEGORY_NAMES = ['__background__', 'eutar_bla', 'heter_ech', 'eutar_tra']
CLASSES_NUMBER = len(INSTANCE_CATEGORY_NAMES)
KEYPOINTS = 7
LOSS_WEIGHTS = {
    "loss_classifier": 1.0,
    "loss_box_reg": 0.7,
    "loss_keypoint": 2.0,
    "loss_objectness": 1.0,
    "loss_rpn_box_reg": 0.7
}

# INFERENCE HYPER-PARAMETERS
BOX_NMS_THRESH = 0.5
RPN_SCORE_THRESH = 0.8
BOX_SCORE_THRESH = 0.8
