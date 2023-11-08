from pprint import pprint

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import keypoints_detector.config as cfg
from keypoints_detector.kpt_rcnn.dataset import create_dataloaders
from keypoints_detector.kpt_rcnn.model import KeypointDetector
from keypoints_detector.kpt_rcnn.utils import calc_oks
from keypoints_detector.kpt_rcnn.utils import tensor2rgb


def prepare_annotation_dict(annotation_dict):
    return dict(boxes=annotation_dict["boxes"].cpu().detach().type(torch.float),
                labels=annotation_dict["labels"].cpu().detach(),
                keypoints=annotation_dict["keypoints"].cpu().detach())


def run_benchmark():
    dataloader = create_dataloaders(images_dir=cfg.IMAGES_PATH, batch_size=1)["test"]
    model = KeypointDetector(tiling=True)
    results, predictions, targets, img_sizes = test(model, dataloader, tile_detector=True, class_metrics=True)

    pprint(results)


def test(model, dataloader_test, device=cfg.DEVICE, tile_detector=False, class_metrics=False):
    mAP = MeanAveragePrecision(class_metrics=class_metrics)
    predictions, targets, img_sizes = [], [], []

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            for img, target in zip(*data):
                predicted = model(tensor2rgb([img])) if tile_detector else model([img.to(device)])[0]
                img_sizes.append(img.shape[:2])

                predicted_dict = prepare_annotation_dict(predicted)
                predicted_dict["scores"] = predicted["scores"].cpu().detach()
                predictions.append(predicted_dict)
                targets.append(prepare_annotation_dict(target))

    mAP.update(predictions, targets)
    results = mAP.compute()
    results["oks"] = calc_oks(predictions, targets, img_sizes)

    return results, predictions, targets, img_sizes


if __name__ == "__main__":
    run_benchmark()
