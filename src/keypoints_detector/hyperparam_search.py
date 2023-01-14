#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import torch
from ray import tune

import keypoints_detector.config as cfg
from keypoints_detector.dataset import create_dataloaders
from keypoints_detector.model import keypoint_detector
from keypoints_detector.predict import run_testing
from keypoints_detector.train import train_one_batch, validate
from keypoints_detector.utils import set_reproducibility_params, create_losses_dict

set_reproducibility_params()
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"


def check_training_params(config):
    device = cfg.DEVICE
    model = keypoint_detector()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"],
                                momentum=config["momentum"])

    dataloaders = create_dataloaders(images_dir=str(cfg.IMAGES_PATH), annotation_file=str(cfg.ANNOTATON_FILE_PATH),
                                     val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO)

    losses_names, _ = create_losses_dict()
    model.train()
    for epoch in range(config["epochs"]):
        epoch_losses = {key: [] for key in losses_names}

        for i, (imgs, targets) in enumerate(dataloaders["train"]):
            train_one_batch(model, device, imgs, targets, optimizer, None, epoch_losses, losses_names)

        with torch.no_grad():
            for i, (imgs, targets) in enumerate(dataloaders["val"]):
                validate(model, device, imgs, targets, epoch_losses, losses_names)

        report_losses = {loss_key: np.asarray(epoch_losses[loss_key]).mean() for loss_key in losses_names}
        tune.report(**report_losses)


def check_inference_params(config):
    test_results = run_testing(images_path=cfg.IMAGES_PATH, annotation_path=cfg.ANNOTATON_FILE_PATH,
                               model_path=cfg.MODEL_PATH, model_config=config)
    tune.report(map_50=test_results["map_50"], oks=test_results["oks"])


def search_hyperparameters(search_mode="training"):
    results_path = Path("/tarmass/src/keypoints_detector/hyperparam_results")
    results_path.mkdir(exist_ok=True)

    if search_mode == "training":
        config = {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
            "momentum": tune.uniform(0, 1),
            "gamma": tune.uniform(0, 1),
            "epochs": 20
        }

        analysis = tune.run(
            check_training_params,
            config=config,
            metric="val_loss_total",
            mode="min",
            num_samples=100,
            resources_per_trial={"cpu": 16, "gpu": 1},
            verbose=1,
        )

    elif search_mode == "inference":
        config = {
            "box_nms_thresh": tune.uniform(0, 1),
        }

        analysis = tune.run(
            check_inference_params,
            config=config,
            metric="oks",
            mode="max",
            num_samples=100,
            resources_per_trial={"cpu": 1, "gpu": 1},
            verbose=1,
        )

    else:
        raise NotImplementedError

    result_df = analysis.best_result_df
    result_df.to_csv(results_path / f"{search_mode}_hyperparams_results.csv")


if __name__ == '__main__':
    search_hyperparameters()
