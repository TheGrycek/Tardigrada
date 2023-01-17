#!/usr/bin/env python3

import warnings
from pathlib import Path
from time import time, strftime, gmtime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter

import keypoints_detector.config as cfg
from keypoints_detector.dataset import create_dataloaders
from keypoints_detector.model import keypoint_detector
from keypoints_detector.predict import test
from keypoints_detector.utils import set_reproducibility_params, create_losses_dict

set_reproducibility_params()
writer = SummaryWriter("./runs/board_results")


def train_one_batch(model, device, imgs, targets, optimizer, scheduler, epoch_losses, losses_names):
    img_batch = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

    loss_dict = model(img_batch, targets)
    loss_total = sum(loss * cfg.LOSS_WEIGHTS[name] for name, loss in loss_dict.items())

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    epoch_losses["train_loss_total"].append(float(loss_total.item()))
    for loss_key in losses_names[1: 6]:
        epoch_losses[loss_key].append(float(loss_dict[loss_key.replace("train_", "")].item()))


def validate(model, device, imgs, targets, epoch_losses, losses_names):
    img_batch = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

    val_loss_dict = model(img_batch, targets)
    val_loss_total = float(sum(loss * cfg.LOSS_WEIGHTS[name] for name, loss in val_loss_dict.items()).item())

    epoch_losses["val_loss_total"].append(val_loss_total)
    for loss_key in losses_names[7:]:
        epoch_losses[loss_key].append(float(val_loss_dict[loss_key.replace("val_", "")].item()))

    return val_loss_total


def add_tensorboard_image(img_tensor):
    global writer
    grid = torchvision.utils.make_grid([img_tensor[0]])
    writer.add_image("tardigrada input", grid)
    writer.close()


def add_tensorboard_scalars(epoch_metrics, epoch):
    global writer
    for name, value in epoch_metrics.items():
        writer.add_scalar(name, value, epoch)
    writer.close()


def save_plots_plt(losses):
    out_path = Path("./training_results")
    out_path.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(50, 20))
    for i, (loss_key, loss_list) in enumerate(losses.items(), 1):
        plt.subplot(2, 6, i)
        plt.grid(True)
        plt.plot([i + 1 for i in range(len(loss_list))], loss_list)
        plt.title(loss_key, fontsize=30)
        plt.xlabel("Epoch", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.yticks(fontsize=17)
        plt.xticks(fontsize=17)

    plt.savefig(out_path / "training_results")


def train(images_path=cfg.IMAGES_PATH, annotation_path=cfg.ANNOTATION_FILE_PATH, device=cfg.DEVICE,
          save_plots_matplotlib=True, print_stats_n_epoch=1, export_onnx=True, save_best=True):
    model = keypoint_detector()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)

    dataloaders = create_dataloaders(images_dir=str(images_path),
                                     annotation_file=str(annotation_path),
                                     val_ratio=cfg.VAL_RATIO,
                                     test_ratio=cfg.TEST_RATIO)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=cfg.MILESTONES,
                                         gamma=cfg.GAMMA)

    img_example, target_example = (iter(dataloaders["val"])).next()
    add_tensorboard_image(img_example)

    losses_names, losses = create_losses_dict()

    start_total = time()
    best_val_loss = float("inf")
    for epoch in range(cfg.EPOCHS):
        start_epoch = time()
        model.train()
        epoch_losses = {key: [] for key in losses_names}

        for i, (imgs, targets) in enumerate(dataloaders["train"]):
            train_one_batch(model, device, imgs, targets, optimizer, scheduler, epoch_losses, losses_names)

        epoch_mean_val_loss = 0
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(dataloaders["val"]):
                val_loss_total = validate(model, device, imgs, targets, epoch_losses, losses_names)
                epoch_mean_val_loss += val_loss_total

        epoch_mean_val_loss /= len(dataloaders["val"])
        if save_best and epoch_mean_val_loss < best_val_loss:
            best_val_loss = epoch_mean_val_loss
            torch.save(model.state_dict(), f"checkpoints/keypoints_detector_best.pth")

        for loss_key in losses_names:
            losses[loss_key].append(np.asarray(epoch_losses[loss_key]).mean())

        if cfg.CHECKPOINT_SAVE_INTERVAL:
            if epoch % cfg.CHECKPOINT_SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), f"checkpoints/keypoints_detector{epoch}.pth")

        model.eval()
        test_metrics = test(model, dataloaders["val"])
        add_tensorboard_scalars({key: value[-1] for key, value in epoch_losses.items()}, epoch)
        add_tensorboard_scalars(test_metrics, epoch)

        if (epoch + 1) % print_stats_n_epoch == 0:
            print(f"epoch: {epoch + 1}, "
                  f"training loss={epoch_losses['train_loss_total'][-1]:.5f}, "
                  f"validation loss={epoch_losses['val_loss_total'][-1]:.5f}, "
                  f"map_50={test_metrics['map_50']:.5f}, "
                  f"oks={test_metrics['oks']:.5f}, "
                  f" time: {time() - start_epoch}")

    torch.save(model.state_dict(), "checkpoints/keypoints_detector_last.pth")
    print(f"TOTAL TRAINING TIME: {strftime('%H:%M:%S', gmtime(time() - start_total))}")

    if save_plots_matplotlib:
        save_plots_plt(losses)

    # TODO: fix img_example shape
    # if export_onnx:
    #     torch.onnx.export(model, img_example, "checkpoints/keypoint_rcnn.onnx", verbose=True,
    #                       input_names=["input"], output_names=["output"])

    return model


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    print(f"MY DEVICE: {cfg.DEVICE}")
    train()
