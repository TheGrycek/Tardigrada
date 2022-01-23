import random
from pathlib import Path

import cv2
import numpy as np
import torch

import config as cfg
from dataset import load_data
from model import segmenter

import matplotlib.pyplot as plt  # TODO: matplotlib must be imported after torchvision model to avoid SIGSEGV error!


def colour_mask(mask):
    mask = mask[0]
    mask[mask >= 0.2] = 255
    mask = mask.astype(np.uint8)
    channels = [np.ones_like(mask) * random.randrange(0, 255, 1) for _ in range(3)]

    mask_colour = np.stack(channels, axis=2)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return cv2.bitwise_and(mask, mask_colour)


def train(device, checkpoint_save_interval=True, save_plots=True):
    model = segmenter()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)

    dataloader = load_data(images_dir="../images",
                           annotation_file="../labels_tardigrada_2022-01-07-20-11-35-797274.json")

    losses = {"loss_total": [],
              "loss_classifier": [],
              "loss_box_reg": [],
              "loss_mask": [],
              "loss_objectness": [],
              "loss_rpn_box_reg": []}

    for epoch in range(cfg.EPOCHS):
        for i, (img, targets) in enumerate(dataloader):
            img = [img.to(device) for img in img]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(img, targets)
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

    # torch.save(model.state_dict(), f"checkpoints/segmenter.pth")

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


def predict(img, device):
    model = segmenter()
    model.load_state_dict(torch.load("./checkpoints/segmenter.pth"))
    model.eval().to(device)

    with torch.no_grad():
        img_shape = img.shape
        input_tensor = torch.from_numpy(img.astype(np.float32) / 255).to(device)
        input_tensor = input_tensor.view([1, 3, img_shape[0], img_shape[1]])
        predicted = model(input_tensor)[0]
        bboxes = predicted["boxes"].cpu().detach().numpy()
        labels = predicted["labels"].cpu().detach().numpy().astype(np.uint8)
        scores = predicted["scores"].cpu().detach().numpy()
        masks = predicted['masks'].cpu().detach().numpy()
        print(f"SCORES: {scores}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
            rgb_mask = colour_mask(masks[i])

            img = cv2.addWeighted(img, 1, rgb_mask, 0.9, 0)

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
    img = cv2.imread("../images/krio5_OM_1.5_1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    train(cfg.DEVICE, checkpoint_save_interval=False)
    # predict(img, cfg.DEVICE)
