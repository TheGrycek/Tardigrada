from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as cfg
from dataset import load_data, get_contours_labels
from model import PoseNet


def train(dataset_path, device, model, checkpoint_save_interval=None, save_plots=True):
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE)
    dataloader = load_data(dataset_path,
                           test_ratio=cfg.TEST_RATIO,
                           batch_size=cfg.BATCH_SIZE,
                           transform=True,
                           test=False,
                           shuffle=False)
    loss_epochs = []

    for epoch in range(cfg.EPOCHS):
        epoch_loss = []
        for i, (data_sample, label) in enumerate(dataloader):
            input_tensor = data_sample.to(device)
            label_tensor = label.to(device)
            predicted = model(input_tensor)

            loss = cfg.LOSS_FUNCTION(label_tensor, predicted)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss.append(loss.item())

        if checkpoint_save_interval:
            if epoch % checkpoint_save_interval == 0:
                torch.save(model.state_dict(), f"checkpoints/pose_net_checkpoint{epoch}.pth")

        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch + 1}, loss={loss.item()}:.4f")

        loss_epochs.append(np.mean(epoch_loss))  # mean epoch loss

    if save_plots:
        out_path = Path("./training_results")
        out_path.mkdir(exist_ok=True, parents=True)
        plt.plot([i + 1 for i in range(len(loss_epochs))], loss_epochs)
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.savefig(out_path / "loss")
        plt.show()

    return model


def predict(dataset_path, device, model=None):
    if model == None:
        model = PoseNet().to(device)
        model.load_state_dict(torch.load("./checkpoints/pose_net.pth"))

    with torch.no_grad():
        dataloader = load_data(dataset_path, transform=False, shuffle=False, batch_size=1,
                               test=False, test_ratio=cfg.TEST_RATIO)
        contours = get_contours_labels(dataset_path)
        for i, (data_sample, label) in enumerate(dataloader):
            input_tensor = data_sample.to(device)
            input_tensor = input_tensor.view([1, 3, 227, 227])
            predicted = model(input_tensor).cpu()
            predicted = predicted.view([4, 2])

            print(f"PREDICTED: {predicted}")
            single_label = label.view([4, 2])
            contour = contours[i]
            image = data_sample.view([227, 227, 3]).numpy()
            image = image.astype(np.uint8)
            for pt in predicted:
                x, y = pt
                x = round(x.item() * 227)
                y = round(y.item() * 227)
                cv2.circle(image, (x, y), radius=2, thickness=2, color=(0, 0, 255))

            cv2.imshow("image_predict", image)
            cv2.waitKey(1000)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(contour[:, 0], contour[:, 1], color="red")
            ax.scatter(predicted[:, 0], predicted[:, 1], color="green", s=80)
            ax.scatter(single_label[:, 0], single_label[:, 1], color="blue", s=80)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.show()


if __name__ == '__main__':
    print(f"Used device: {cfg.DEVICE}")
    dataset_path = Path("../images/dataset")

    model = PoseNet().to(cfg.DEVICE)
    print(f"Network architecture: {model.layers}")
    trained_net = train(dataset_path, cfg.DEVICE, model, checkpoint_save_interval=cfg.CHECKPOINT_SAVE_INTERVAL)
    torch.save(trained_net.state_dict(), "checkpoints/pose_net.pth")
    print("Training finished")

    predict(dataset_path, cfg.DEVICE, trained_net)
