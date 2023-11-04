#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from ray import tune

import keypoints_detector.config as cfg
from keypoints_detector.kpt_rcnn.dataset import create_dataloaders
from keypoints_detector.kpt_rcnn.model import keypoint_detector, KeypointDetector
from keypoints_detector.kpt_rcnn.test import test
from keypoints_detector.kpt_rcnn.train import train_one_batch, validate
from keypoints_detector.kpt_rcnn.utils import set_reproducibility_params, create_losses_dict

set_reproducibility_params()
# os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--search_mode", type=str, default="inference",
                        help="Hyperparameters tuning mode- inference/training")

    return parser.parse_args()


def check_training_params(config):
    device = cfg.DEVICE
    model = keypoint_detector()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"],
                                momentum=config["momentum"])

    dataloaders = create_dataloaders(images_dir=str(cfg.IMAGES_PATH),
                                     annotation_file=str(cfg.ANNOTATION_FILE_PATH),
                                     val_ratio=cfg.VAL_RATIO,
                                     test_ratio=cfg.TEST_RATIO)

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
    model = KeypointDetector(model_path=cfg.RCNN_MODEL_PATH, **config)
    dataloaders = create_dataloaders(images_dir=str(cfg.IMAGES_PATH),
                                     annotation_file=str(cfg.ANNOTATION_FILE_PATH),
                                     val_ratio=cfg.VAL_RATIO,
                                     test_ratio=cfg.TEST_RATIO)

    test_results = test(model, dataloaders["test"], cfg.DEVICE, tile_detector=True)[0]
    tune.report(map_50=test_results["map_50"], oks=test_results["oks"])


def search_hyperparameters(args):
    results_path = Path(f"{cfg.REPO_ROOT}/src/keypoints_detector/kpt_rcnn")
    results_path.mkdir(exist_ok=True)

    if args.search_mode == "training":
        config = {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
            "momentum": tune.uniform(0, 1),
            "gamma": tune.uniform(0, 1),
            "epochs": 20
        }
        metric = {
            "metric": "val_loss_total",
            "mode": "min"
        }
        test_function = check_training_params

    elif args.search_mode == "inference":
        config = {
            "rpn_score_thresh": tune.uniform(0.0, 0.9),
            "box_score_thresh": tune.uniform(0.0, 0.9),
            "box_nms_thresh": tune.uniform(0.0, 0.9),
            "tiling_nms_thresh": tune.uniform(0.1, 0.9),
        }
        metric = {
            "metric": "map_50",
            "mode": "max"
        }
        test_function = check_inference_params

    else:
        raise NotImplementedError

    analysis = tune.run(
        test_function,
        **metric,
        config=config,
        num_samples=100,
        resources_per_trial={"cpu": 16, "gpu": 1},
        verbose=1,
    )

    # TODO: check why best result isn't working
    result_df = analysis.best_result_df
    result_df.to_csv(results_path / f"{args.search_mode}_hyperparams_results.csv")


if __name__ == '__main__':
    search_hyperparameters(parse_args())
