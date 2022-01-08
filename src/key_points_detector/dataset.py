import glob
import json

import albumentations as alb
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# from albumentations.pytorch import ToTensorV2  # TODO resolve SIGSEGV error


class PoseKeypointsDatset(Dataset):
    def __init__(self, dataset_dir, img_size=227, transform=True, colour=True):
        super().__init__()
        self.img_size = img_size
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.colour = colour
        self.img_shape = 0

        self.images, self.labels = self.load_labels()
        self.n_samples = self.images.shape[0]

    def __getitem__(self, item):
        image = self.images[item]
        labels = self.labels[item] * self.img_size
        labels = np.round_(labels, decimals=0)

        if self.transform:
            augmentations = self.augment(image, labels)
            image = augmentations["image"]
            labels = augmentations["keypoints"]

        image = torch.from_numpy(image.astype(np.float32)).reshape(3 if self.colour else 1,
                                                                   self.img_shape, self.img_shape)
        labels = np.array(labels, dtype=np.float32) / self.img_size
        labels = torch.from_numpy(labels.flatten())

        return image, labels

    def __len__(self):
        return self.n_samples

    def load_labels(self):
        files = glob.glob(str(self.dataset_dir) + '/*.json')
        images, labels_points = [], []

        for i, file in enumerate(files):
            data_dict = json.load(open(file, "r"))
            points = np.array(data_dict["key_points"], dtype=np.float32)
            ratio, image = resize_pad(cv2.imread(str(file)[:-4] + "png", 1 if self.colour else 0))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels_points.append(points)
            images.append(image.astype(np.uint8))

            if self.img_shape == 0:
                self.img_shape = image.shape[0]

        return np.array(images), np.array(labels_points)

    @staticmethod
    def augment(img, key_points):

        one_of = [alb.ImageCompression(p=0.8),
                  alb.Blur(blur_limit=5, p=0.8),
                  alb.GaussNoise(p=0.8),
                  alb.CLAHE(p=0.8),
                  alb.RandomGamma(p=0.8)]

        transform = alb.Compose([
            # alb.HorizontalFlip(p=0.5),  # This augmentation will affect labels (incorrect side left/right)!
            # alb.VerticalFlip(p=0.5),
            # alb.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, p=0.8),  # uncomment after dataset increase
            alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            alb.OneOf(one_of, p=0.8),
            alb.OneOf(one_of, p=0.8),
            # alb.normalize(mean=[], std=[], max_pixel_value=255)  # TODO: calculate dataset normalize params
            # ToTensorV2(),  # TODO: resolve import issue
        ], keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False))

        return transform(image=img, keypoints=key_points)


def resize_pad(image, img_size=None, new_size=227, get_scales_only=False):
    if image is not None:
        img_size = image.shape[:2]

    ratio = float(new_size) / max(img_size)
    new_shape = [int(im * ratio) for im in img_size]

    pad_x = new_size - new_shape[1]
    pad_y = new_size - new_shape[0]
    left, top = pad_x // 2, pad_y // 2
    right, bottom = pad_x - (pad_x // 2), pad_y - (pad_y // 2)

    if get_scales_only:
        return ratio, (left, right, top, bottom)

    resized_img = cv2.resize(image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)

    return ratio, cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def get_contours_labels(dataset_dir):
    files = glob.glob(str(dataset_dir) + '/*.json')
    contours_points = []
    for file in files:
        data_dict = json.load(open(file, "r"))
        contours_points.append(np.array(data_dict["contour_points"], dtype=np.float32))

    return contours_points


def load_data(dataset_path, batch_size=4, transform=True, shuffle=False, num_workers=0,
              train_test_split=False, test_ratio=0.1):
    dataset = PoseKeypointsDatset(dataset_path, transform=transform)

    if train_test_split:
        if test_ratio > 0:
            dataset_len = len(dataset)
            torch.manual_seed(1)
            indices = torch.randperm(dataset_len).tolist()
            dataset = torch.utils.data.Subset(dataset, indices[:-int(np.ceil(dataset_len * test_ratio))])
            dataset_test = torch.utils.data.Subset(dataset, indices[int(-np.ceil(dataset_len * test_ratio)):])
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers)
            return dataloader, dataloader_test

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader, None
