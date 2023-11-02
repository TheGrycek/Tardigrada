#!/usr/bin/env python3
import warnings
from argparse import ArgumentParser
from pathlib import Path
from time import time, strftime, gmtime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import keypoints_detector.config as cfg
from keypoints_detector.dataset import create_dataloaders
from keypoints_detector.model import keypoint_detector
from keypoints_detector.test import test
from keypoints_detector.utils import set_reproducibility_params, create_losses_dict

set_reproducibility_params()
writer = SummaryWriter("./runs/board_results")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-img_dir", "--images_directory", type=str, default=cfg.IMAGES_PATH,
                        help="Dataset images directory.")
    parser.add_argument("-annot_dir", "--annotation_directory", type=str, default=cfg.ANNOTATION_FILE_PATH,
                        help="Annotation json file directory.")
    parser.add_argument("-d", "--device", type=str, default=cfg.DEVICE, choices=["cpu", "cuda"],
                        help="Device: cpu or cuda.")
    parser.add_argument("-save_plt", "--save_plots_matplotlib", type=bool, default=True,
                        help="Save plots with training and validation losses as an image, using matplotlib.pyplot.")
    parser.add_argument("-n_print", "--print_stats_n_epoch", type=int, default=1,
                        help="Print training stats every n-th epoch.")
    parser.add_argument("-onnx", "--export_onnx", type=bool, default=False,
                        help="Export final model to the ONNX file.")
    parser.add_argument("-best", "--save_best", type=bool, default=True,
                        help="Save separate model with the smallest validation loss.")
    parser.add_argument("-r", "--resume", action="store_true",
                        help="Resume training from the last checkpoint")
    parser.add_argument("-r_dir", "--resume_dir", type=str, default="checkpoints/checkpoint_780.pth",
                        help="Resume checkpoint directory")

    return parser.parse_args()


def send_to_device(imgs, targets, device):
    img_batch = [img.to(device) for img in imgs]
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
    return img_batch, targets


def optimizer_to_device(optim, device):
    def set_tensor_device(param):
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

    for parameter in optim.state.values():
        set_tensor_device(parameter)
        if isinstance(parameter, dict):
            for sub_parameter in parameter.values():
                set_tensor_device(sub_parameter)


def train_one_batch(model, device, imgs, targets, optimizer, scheduler, epoch_losses, losses_names):
    loss_dict = model(*send_to_device(imgs, targets, device))
    if cfg.RANDOM_LOSS_WEIGHTS:
        random_weights = F.softmax(torch.randn(len(cfg.LOSS_WEIGHTS)), dim=-1)
        loss_total = sum(loss * random_weights[i] for i, loss in enumerate(loss_dict.values()))
    else:
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
    val_loss_dict = model(*send_to_device(imgs, targets, device))
    val_loss_total = float(sum(loss * cfg.LOSS_WEIGHTS[name] for name, loss in val_loss_dict.items()).item())

    epoch_losses["val_loss_total"].append(val_loss_total)
    for loss_key in losses_names[7:]:
        epoch_losses[loss_key].append(float(val_loss_dict[loss_key.replace("val_", "")].item()))

    return val_loss_total


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
        epochs_ = np.array([i + 1 for i in range(len(loss_list))])
        losses_ = np.array(loss_list)
        for wrong_val in ("nan", "inf"):
            good_arr = losses_ != float(wrong_val)
            epochs_ = epochs_[good_arr]
            losses_ = losses_[good_arr]

        plt.subplot(2, 6, i)
        plt.grid(True)
        plt.plot(epochs_, losses_)
        plt.title(loss_key, fontsize=30)
        plt.xlabel("Epoch", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.yticks(fontsize=17)
        plt.xticks(fontsize=17)

    plt.savefig(out_path / "training_results")


def freeze_layers(model, grad=False):
    for layer_name in cfg.FREEZE_LAYERS:
        eval(f"model.{layer_name}").requires_grad_(grad)

    return model


def save_checkpoint(epoch, model, optimizer, scheduler, losses, checkpoint_name="keypoints_detector_last.pth"):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'losses': losses,
    }, f"checkpoints/{checkpoint_name}")


def initialize_training_elements(args):
    start_epoch = 0
    model = keypoint_detector()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM,
                                dampening=cfg.DAMPENING,
                                weight_decay=cfg.WEIGHT_DECAY,
                                nesterov=cfg.NESTEROV)

    dataloaders = create_dataloaders(images_dir=args.images_directory,
                                     annotation_file=args.annotation_directory,
                                     val_ratio=cfg.VAL_RATIO,
                                     test_ratio=cfg.TEST_RATIO,
                                     transform_train=cfg.TRANSFORM_TRAIN)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=cfg.MILESTONES,
                                         gamma=cfg.GAMMA)

    losses_names, losses = create_losses_dict()

    if args.resume:
        checkpoint = torch.load(args.resume_dir)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losses = checkpoint["losses"]
        optimizer_to_device(optimizer, args.device)

    model.to(args.device)

    return start_epoch, model, optimizer, dataloaders, scheduler, losses_names, losses


def train(args):
    start_epoch, model, optimizer, dataloaders, scheduler, losses_names, losses = initialize_training_elements(args)

    start_total = time()
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.EPOCHS):
        start_timer = time()
        model.train()
        epoch_losses = {key: [] for key in losses_names}

        if epoch in cfg.UNFREEZE_EPOCHS:
            model = freeze_layers(model, grad=True)

        if epoch in cfg.FREEZE_EPOCHS:
            model = freeze_layers(model, grad=False)

        for i, (imgs, targets) in enumerate(dataloaders["train"]):
            train_one_batch(model, args.device, imgs, targets, optimizer, scheduler, epoch_losses, losses_names)

        epoch_mean_val_loss = 0
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(dataloaders["val"]):
                val_loss_total = validate(model, args.device, imgs, targets, epoch_losses, losses_names)
                epoch_mean_val_loss += val_loss_total

        epoch_mean_val_loss /= len(dataloaders["val"])

        for loss_key in losses_names:
            losses[loss_key].append(np.asarray(epoch_losses[loss_key]).mean())

        if args.save_best and epoch_mean_val_loss < best_val_loss:
            best_val_loss = epoch_mean_val_loss
            save_checkpoint(epoch, model, optimizer, scheduler, losses,
                            checkpoint_name=f"keypoints_detector_best.pth")

        if cfg.CHECKPOINT_SAVE_INTERVAL and epoch % cfg.CHECKPOINT_SAVE_INTERVAL == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, losses,
                            checkpoint_name=f"checkpoint_{epoch}.pth")

        model.eval()
        test_metrics = test(model, dataloaders["val"])[0]
        add_tensorboard_scalars({key: value[-1] for key, value in epoch_losses.items()}, epoch)
        add_tensorboard_scalars(test_metrics, epoch)

        if (epoch + 1) % args.print_stats_n_epoch == 0:
            print(f"epoch: {epoch + 1}, "
                  f"training loss={losses['train_loss_total'][-1]:.5f}, "
                  f"validation loss={losses['val_loss_total'][-1]:.5f}, "
                  f"map_50={test_metrics['map_50']:.5f}, "
                  f"oks={test_metrics['oks']:.10f}, "
                  f" time: {time() - start_timer}")

    save_checkpoint(cfg.EPOCHS - 1, model, optimizer, scheduler, losses)
    print(f"TOTAL TRAINING TIME: {strftime('%H:%M:%S', gmtime(time() - start_total))}")

    if args.save_plots_matplotlib:
        save_plots_plt(losses)

    # TODO: create onnx export
    # if args.export_onnx:
    #     torch.onnx.export(model, img_example, "checkpoints/keypoint_rcnn.onnx", verbose=True,
    #                       input_names=["input"], output_names=["output"])

    return model


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    print(f"MY DEVICE: {cfg.DEVICE}")
    train(parse_args())
