import random
from pathlib import Path
from time import time, strftime, gmtime

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
import config as cfg
from dataset import load_data
from model import keypoint_detector
from collections import OrderedDict

import matplotlib.pyplot as plt  # TODO: matplotlib must be imported after torchvision model to avoid SIGSEGV error!

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

writer = SummaryWriter("./runs/board_results")


def train_one_epoch(model, device, imgs, targets, optimizer, scheduler, epoch_losses, losses_names):
    img_batch = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

    loss_dict = model(img_batch, targets)
    loss_total = sum(loss * cfg.LOSS_WEIGHTS[name] for name, loss in loss_dict.items())

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    scheduler.step()
    epoch_losses["train_loss_total"].append(float(loss_total.item()))
    for loss_key in losses_names[1: 6]:
        epoch_losses[loss_key].append(float(loss_dict[loss_key.replace("train_", "")].item()))


def validate(model, device, imgs, targets, epoch_losses, losses_names):
    img_batch = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

    val_loss_dict = model(img_batch, targets)
    val_loss_total = sum(loss * cfg.LOSS_WEIGHTS[name] for name, loss in val_loss_dict.items())

    epoch_losses["val_loss_total"].append(float(val_loss_total.item()))
    for loss_key in losses_names[7:]:
        epoch_losses[loss_key].append(float(val_loss_dict[loss_key.replace("val_", "")].item()))


def add_tensorboard_image(img_tensor):
    global writer
    grid = torchvision.utils.make_grid([img_tensor[0]])
    writer.add_image("tardigrada input", grid)
    writer.close()


def add_tensorboard_scalars(epoch_losses, epoch):
    global writer
    for loss_name, loss_value in epoch_losses.items():
        writer.add_scalar(loss_name, loss_value[-1], epoch)
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
    plt.show()


def calculate_metrics():
    pass


def train(images_path, annotation_path, device, save_plots_matplotlib=True, print_stats_n_epoch=1, export_onnx=True):
    model = keypoint_detector()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)

    dataloaders = load_data(images_dir=str(images_path),
                            annotation_file=str(annotation_path),
                            transform=False, val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=cfg.GAMMA)

    img_example, target_example = (iter(dataloaders["val"])).next()
    add_tensorboard_image(img_example)

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

    losses = OrderedDict({key: [] for key in losses_names})

    start_total = time()
    model.train()
    for epoch in range(cfg.EPOCHS):
        start_epoch = time()
        epoch_losses = {key: [] for key in losses_names}

        for i, (imgs, targets) in enumerate(dataloaders["train"]):
            train_one_epoch(model, device, imgs, targets, optimizer, scheduler, epoch_losses, losses_names)

        # validate using training mode, since batch-norm layers are frozen
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(dataloaders["val"]):
                validate(model, device, imgs, targets, epoch_losses, losses_names)

        for loss_key in losses_names:
            losses[loss_key].append(np.asarray(epoch_losses[loss_key]).mean())

        if cfg.CHECKPOINT_SAVE_INTERVAL:
            if epoch % cfg.CHECKPOINT_SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), f"checkpoints/segmenter_checkpoint{epoch}.pth")

        if (epoch + 1) % print_stats_n_epoch == 0:
            print(f"epoch: {epoch + 1}, "
                  f"training loss={epoch_losses['train_loss_total'][-1]:.4f}, "
                  f"validation loss={epoch_losses['val_loss_total'][-1]:.4f}, "
                  f" time: {time() - start_epoch}")

        # calculate_metrics()
        add_tensorboard_scalars(epoch_losses, epoch)

    torch.save(model.state_dict(), "checkpoints/keypoints_detector_test.pth")
    print(f"TOTAL TRAINING TIME: {strftime('%H:%M:%S', gmtime(time() - start_total))}")

    if save_plots_matplotlib:
        save_plots_plt(losses)

    # TODO: fix img_example shape
    # if export_onnx:
        # torch.onnx.export(model, img_example, "checkpoints/keypoint_rcnn.onnx", verbose=True,
        #                   input_names=["input"], output_names=["output"])

    return model


if __name__ == '__main__':
    images_path = Path("../images/train")
    annotation_path = Path(cfg.ANNOTATON_FILE)
    print(f"MY DEVICE: {cfg.DEVICE}")
    train(images_path, annotation_path, cfg.DEVICE,
          save_plots_matplotlib=True,
          print_stats_n_epoch=1
          )
