import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import Conv2d, Linear
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt  # TODO: matplotlib must be imported after torchvision model to avoid SIGSEGV error!

import config as cfg
from dataset import SegmentationDataset


def segmenter():
    model = maskrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    torch.manual_seed(1)
    new_conv2d = Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    model.roi_heads.mask_predictor.mask_fcn_logits = new_conv2d

    new_linear = Linear(in_features=1024, out_features=3, bias=True)
    model.roi_heads.box_predictor.cls_score = new_linear

    return model


def random_colour_masks(mask):
    mask = mask.reshape(mask.shape[::-1])

    random.seed(1)
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, len(colours))]

    return np.stack([r, g, b], axis=2)


def train(device, checkpoint_save_interval=True, save_plots=True):
    model = segmenter()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)

    # TODO: create Dataloder
    dataset = SegmentationDataset(images_dir="../images",
                                  annotation_file="../labels_tardigrada_2022-01-07-20-11-35-797274.json")

    losses = {"loss_total": [],
              "loss_classifier": [],
              "loss_box_reg": [],
              "loss_mask": [],
              "loss_objectness": [],
              "loss_rpn_box_reg": []}

    for epoch in range(cfg.EPOCHS):
        for i, (img, targets) in enumerate(dataset):  # only one image at the moment
            img = img.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            loss_dict = model([img], [targets])
            loss_total = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            for loss_key in loss_dict.keys():
                losses[loss_key].append(loss_dict[loss_key].item())

            losses["loss_total"].append(loss_total.item())

        if checkpoint_save_interval:
            if epoch % cfg.CHECKPOINT_SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), f"checkpoints/segmenter_checkpoint{epoch}.pth")

        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch + 1}, loss={loss_total.item()}:.4f")

    torch.save(model.state_dict(), f"checkpoints/segmenter.pth")

    if save_plots:
        out_path = Path("./training_results")
        out_path.mkdir(exist_ok=True, parents=True)

        for loss_key, loss_list in losses.items():
            plt.plot([i + 1 for i in range(len(loss_list))], loss_list)
            plt.title(f"Training {loss_key}")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.savefig(out_path / loss_key)
            plt.show()
            plt.cla()

    return model


def predict(img, device):
    model = segmenter()
    model.load_state_dict(torch.load("./checkpoints/segmenter.pth"))
    model.eval().to(device)

    with torch.no_grad():
        img_shape = img.shape
        input_tensor = torch.from_numpy(img.astype(np.float32)).to(device)
        input_tensor = input_tensor.view([1, 3, img_shape[1], img_shape[0]])
        predicted = model(input_tensor)[0]
        bboxes = predicted["boxes"].cpu().detach().numpy()
        labels = predicted["labels"].cpu().detach().numpy().astype(np.uint8)
        scores = predicted["scores"].cpu().detach().numpy()
        masks = (predicted['masks'] > 0.5).cpu().detach().numpy()

        print(f"MASKS LEN: {len(masks)}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
            rgb_mask = random_colour_masks(masks[i])
            rgb_mask = rgb_mask.reshape(img_shape)
            img = cv2.addWeighted(img, 1, rgb_mask, 0.8, 0)

            pt1 = tuple(bboxes[i][:2].astype(np.uint16))
            pt2 = tuple(bboxes[i][2:].astype(np.uint16))
            cv2.rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=2)

            position = (int(bboxes[i][0]), int(bboxes[i][1]) + 100)
            img = cv2.putText(img, cfg.INSTANCE_CATEGORY_NAMES[labels[i]],
                        position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 0, 0), thickness=2)

        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == '__main__':
    img = cv2.imread("../images/krio5_OM_1.5_3.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    train(cfg.DEVICE, checkpoint_save_interval=False)
    predict(img, cfg.DEVICE)
