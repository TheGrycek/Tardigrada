import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_FUNCTION = torch.nn.MSELoss()
INPUT_IMAGE_SIZE = 227
LEARNING_RATE = 0.01
EPOCHS = 60
BATCH_SIZE = 1
TEST_RATIO = 0
CHECKPOINT_SAVE_INTERVAL = 10
NUM_WORKERS = 0
