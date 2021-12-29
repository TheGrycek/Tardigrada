import glob
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from mass_calculation import calculate_mass


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 8))

    def forward(self, x):
        return self.layers(x)


class TardigrdeDatset(Dataset):
    def __init__(self, dataset_dir, test=True, test_ratio=0.1, colour=True):
        self.dataset_dir = dataset_dir
        self.colour = colour
        self.test = test
        self.test_ratio = test_ratio

        images, labels_points = self.load_labels()
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels_points)

        self.n_samples = images.shape[0]

    def __getitem__(self, item):
        images = self.images[item].view([3 if self.colour else 1, self.img_shape, self.img_shape])
        labels = self.labels[item]
        return images, labels

    def __len__(self):
        return self.n_samples

    def load_labels(self):
        files = glob.glob(str(self.dataset_dir) + '/*.json')
        test_files_number = round(len(files) * self.test_ratio)
        images, labels_points = [], []
        img_shape = 0
        for i, file in enumerate(files):
            if self.test:
                if i >= test_files_number:
                    break
            else:
                if i < test_files_number:
                    continue
            data_dict = json.load(open(file, "r"))
            points = np.array(data_dict["key_points"], dtype=np.float32).flatten()
            image = cv2.imread(str(file)[:-4] + "png", 1 if self.colour else 0)
            labels_points.append(points)
            images.append(image.astype(np.float32))

            if img_shape == 0:
                self.img_shape = image.shape[0]

        return np.array(images), np.array(labels_points)


def load_data(dataset_path, batch_size=4, shuffle=False, num_workers=2, test=False, test_ratio=0.1):
    dataset = TardigrdeDatset(dataset_path, test=test, test_ratio=test_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


def get_contours_labels(dataset_dir):
    files = glob.glob(str(dataset_dir) + '/*.json')
    contours_points = []
    for file in files:
        data_dict = json.load(open(file, "r"))
        contours_points.append(np.array(data_dict["contour_points"], dtype=np.float32))

    return contours_points


def train(dataset_path, device, model, checkpoints_save_interval=None, save_plots=True):
    loss_function = nn.MSELoss()
    learning_rate = 0.01
    epochs = 50
    batch_size = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    dataloader = load_data(dataset_path, test=False, test_ratio=0.1, shuffle=False, batch_size=batch_size)
    loss_epochs = []

    for epoch in range(epochs):
        epoch_loss = []
        for i, (data_sample, label) in enumerate(dataloader):
            input_tensor = data_sample.to(device)
            label_tensor = label.to(device)
            predicted = model(input_tensor)

            loss = loss_function(label_tensor, predicted)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss.append(loss.item())

        if checkpoints_save_interval:
            if epoch % checkpoints_save_interval == 0:
                torch.save(model.state_dict(), f"checkpoints/pose_net_checkpoint{epoch}.pth")

        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch + 1}, loss={loss.item()}:.4f")

        loss_epochs.append(np.mean(epoch_loss))

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
        total_mass = 0
        dataloader = load_data(dataset_path, shuffle=False, batch_size=1, test=True, test_ratio=0.1)
        contours = get_contours_labels(dataset_path)
        for i, (data_sample, label) in enumerate(dataloader):
            input_tensor = data_sample.to(device)
            input_tensor = input_tensor.view([1, 3, 227, 227])
            predicted = model(input_tensor).cpu()
            predicted = predicted.view([4, 2])

            print(f"PREDICTED: {predicted}")
            single_label = label.view([4, 2])
            mass = calculate_mass(predicted.numpy())
            total_mass += mass
            contour = contours[i]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(contour[:, 0], contour[:, 1], color="red")
            ax.scatter(predicted[:, 0], predicted[:, 1], color="green", s=80)
            ax.scatter(single_label[:, 0], single_label[:, 1], color="blue", s=80)
            plt.show()

        print(f"Total mass: {total_mass}")


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Used device: {device}")
    dataset_path = Path("../images/dataset")

    net = PoseNet().to(device)
    print(f"Network architecture: {net.layers}")
    trained_net = train(dataset_path, device, net, checkpoints_save_interval=0)
    torch.save(trained_net.state_dict(), "checkpoints/pose_net.pth")
    print("Training finished")

    predict(dataset_path, device, trained_net)
