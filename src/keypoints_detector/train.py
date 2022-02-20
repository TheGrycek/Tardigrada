import random
from pathlib import Path

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

import config as cfg
from dataset import load_data
from model import keypoint_detector

import matplotlib.pyplot as plt  # TODO: matplotlib must be imported after torchvision model to avoid SIGSEGV error!

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def validate():
    return


def train(images_path, annotation_path, device, checkpoint_save_interval=True, save_plots=True):
    model = keypoint_detector(cfg.CLASSES_NUMBER, cfg.KEYPOINTS)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=cfg.GAMMA)

    dataloader = load_data(images_dir=str(images_path),
                           annotation_file=str(annotation_path),
                           transform=True)

    losses_names = ["loss_total",
                    "loss_classifier",
                    "loss_box_reg",
                    "loss_keypoint",
                    "loss_objectness",
                    "loss_rpn_box_reg"]

    losses = {key: [] for key in losses_names}

    for epoch in range(cfg.EPOCHS):
        epoch_losses = {key: [] for key in losses_names}

        for i, (img, targets) in enumerate(dataloader):
            img = [img.to(device) for img in img]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(img, targets)
            loss_total = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # val_loss = validate()
            # TODO: Create validate function
            scheduler.step()

            for loss_key in losses_names[1:]:
                epoch_losses[loss_key].append(loss_dict[loss_key].item())

            epoch_losses["loss_total"].append(loss_total.item())

        for loss_key in losses_names:
            losses[loss_key].append((np.asarray(epoch_losses[loss_key]).mean()))

        if checkpoint_save_interval:
            if epoch % cfg.CHECKPOINT_SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), f"checkpoints/segmenter_checkpoint{epoch}.pth")

        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch + 1}, loss={loss_total.item()}:.4f")

    torch.save(model.state_dict(), f"checkpoints/keypoints_detector.pth")

    if save_plots:
        out_path = Path("./training_results")
        out_path.mkdir(exist_ok=True, parents=True)

        plt.figure(figsize=(20, 10))
        for i, (loss_key, loss_list) in enumerate(losses.items(), 1):
            plt.subplot(2, 3, i)
            plt.grid(True)
            plt.plot([i + 1 for i in range(len(loss_list))], loss_list)
            plt.title(f"Training {loss_key}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

        plt.savefig(out_path / "training_results")
        plt.show()

    return model


if __name__ == '__main__':
    images_path = Path("../images")
    annotation_path = Path("../Annotacja_1.json")
    train(images_path, annotation_path, cfg.DEVICE, checkpoint_save_interval=False)
