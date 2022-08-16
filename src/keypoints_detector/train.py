import random
from pathlib import Path
from time import time, strftime, gmtime

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


def train_one_epoch(model, device, imgs, targets, optimizer, scheduler, epoch_losses, losses_names):
    img_batch = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

    loss_dict = model(img_batch, targets)
    loss_total = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()
    epoch_losses["train_loss_total"].append(float(loss_total.item()))
    for loss_key in losses_names[1: 6]:
        epoch_losses[loss_key].append(float(loss_dict[loss_key.replace("train_", "")].item()))

    return epoch_losses, loss_total


def validate(model, device, imgs, targets, epoch_losses, losses_names):
    img_batch = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

    val_loss_dict = model(img_batch, targets)
    val_loss_total = sum(loss for loss in val_loss_dict.values())

    epoch_losses["val_loss_total"].append(float(val_loss_total.item()))
    for loss_key in losses_names[7:]:
        epoch_losses[loss_key].append(float(val_loss_dict[loss_key.replace("val_", "")].item()))

    return epoch_losses, val_loss_total


def train(images_path, annotation_path, device, checkpoint_save_interval=True, save_plots=True, print_stats_n_epoch=1):
    model = keypoint_detector()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)

    dataloaders = load_data(images_dir=str(images_path),
                            annotation_file=str(annotation_path),
                            transform=True, transform_val=False, val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=cfg.GAMMA)

    losses_names = [
        "train_loss_total",
        "train_loss_classifier",
        "train_loss_box_reg",
        "train_loss_keypoint",
        "train_loss_objectness",
        "train_loss_rpn_box_reg",
        "val_loss_total",
        "val_loss_classifier",
        "val_loss_box_reg",
        "val_loss_keypoint",
        "val_loss_objectness",
        "val_loss_rpn_box_reg"
    ]

    losses = {key: [] for key in losses_names}

    start = time()
    model.train()
    for epoch in range(cfg.EPOCHS):
        epoch_losses = {key: [] for key in losses_names}
        for i, (imgs, targets) in enumerate(dataloaders["train"]):
            epoch_losses, train_loss_total = train_one_epoch(model, device, imgs, targets, optimizer, scheduler,
                                                             epoch_losses, losses_names)

        # validate using training mode, since batch-norm layers are frozen
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(dataloaders["val"]):
                epoch_losses, val_loss_total = validate(model, device, imgs, targets, epoch_losses, losses_names)

        for loss_key in losses_names:
            losses[loss_key].append(np.asarray(epoch_losses[loss_key]).mean())

        if checkpoint_save_interval:
            if epoch % cfg.CHECKPOINT_SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), f"checkpoints/segmenter_checkpoint{epoch}.pth")

        if (epoch + 1) % print_stats_n_epoch == 0:
            print(f"epoch: {epoch + 1}, training loss={train_loss_total.item():.4f}, "
                  f"validation loss={val_loss_total.item():.4f}")

    torch.save(model.state_dict(), f"checkpoints/keypoints_detector.pth")
    print(f"TOTAL TRAINING TIME: {strftime('%H:%M:%S', gmtime(time() - start))}")

    if save_plots:
        out_path = Path("./training_results")
        out_path.mkdir(exist_ok=True, parents=True)

        plt.figure(figsize=(40, 20))
        for i, (loss_key, loss_list) in enumerate(losses.items(), 1):
            plt.subplot(2, 6, i)
            plt.grid(True)
            plt.plot([i + 1 for i in range(len(loss_list))], loss_list)
            plt.title(loss_key)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

        plt.savefig(out_path / "training_results")
        plt.show()

    return model


if __name__ == '__main__':
    images_path = Path("../images/train")
    annotation_path = Path("../coco-1659778596.546996.json")
    train(images_path, annotation_path, cfg.DEVICE, checkpoint_save_interval=True)
