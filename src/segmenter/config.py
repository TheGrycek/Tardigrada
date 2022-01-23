import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_FUNCTION = torch.nn.MSELoss()
LEARNING_RATE = 0.01
MOMENTUM = 0.09
WEIGHT_DECAY = 0.0005
EPOCHS = 100
BATCH_SIZE = 1
TEST_RATIO = 0
CHECKPOINT_SAVE_INTERVAL = 10
NUM_WORKERS = 0
INSTANCE_CATEGORY_NAMES = ['__background__', 'scale', 'tardigrade', 'tardigrade_echiniscus']
CLASSES_NUMBER = len(INSTANCE_CATEGORY_NAMES)
