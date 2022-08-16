import random
from pathlib import Path

import albumentations as alb
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from config import KEYPOINTS

import config as cfg


class KeypointsDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=True, transforms_val=False):
        super().__init__()
        self.dataset_dir = Path(images_dir)
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.transforms_val = transforms_val
        self.points_num = KEYPOINTS
        self.dataset_mean = [0.466063529253006, 0.5127472281455994, 0.490399032831192]
        self.dataset_std = [0.08915058523416519, 0.09907367825508118, 0.096004918217659]
        self.max_pixel_value = 255
        self.min_area = 100

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        img_path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(str(self.dataset_dir / img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        objects_num = len(coco_annotation)

        bboxes, areas, keypoints, keypoints_vis, labels, iscrowd = [], [], [], [], [], []
        for i in range(objects_num):
            coco_elem = coco_annotation[i]
            x_min = coco_elem['bbox'][0]
            y_min = coco_elem['bbox'][1]
            x_max = x_min + coco_elem['bbox'][2]
            y_max = y_min + coco_elem['bbox'][3]

            coco_keypoints = coco_elem['keypoints']
            coco_labels = coco_elem["category_id"]
            boxes = [x_min, y_min, x_max, y_max]

            bboxes.append(boxes)
            keypoints.append(coco_keypoints)
            labels.append(coco_labels)
            areas.append(coco_elem['area'])
            iscrowd.append(coco_elem["iscrowd"])

        if self.transforms:
            keypoints = np.asarray(keypoints).reshape((-1, self.points_num, 3))
            visibility = keypoints[:, :, 2].reshape((-1, self.points_num, 1))
            keypoints_no_vis = keypoints[:, :, :2].reshape((-1, 2))
            transformed = self.augment(img, key_points=keypoints_no_vis, labels=labels, bboxes=bboxes)
            keypoints = np.asarray(transformed["keypoints"]).reshape((-1, self.points_num, 2))
            keypoints = np.concatenate((keypoints, visibility), axis=2)
            transformed_bboxes = np.asarray(transformed['bboxes'])
            if np.all(transformed_bboxes):
                bboxes = transformed_bboxes
                img = transformed["image"]

        elif self.transforms_val:
            keypoints = np.asarray(keypoints).reshape((-1, self.points_num, 3))
            visibility = keypoints[:, :, 2].reshape((-1, self.points_num, 1))
            keypoints_no_vis = keypoints[:, :, :2].reshape((-1, 2))
            transformed = self.augment(img, key_points=keypoints_no_vis, labels=labels, bboxes=bboxes, augment_val=True)
            keypoints = np.asarray(transformed["keypoints"]).reshape((-1, self.points_num, 2))
            keypoints = np.concatenate((keypoints, visibility), axis=2)
            transformed_bboxes = np.asarray(transformed['bboxes'])
            if np.all(transformed_bboxes):
                bboxes = transformed_bboxes
                img = transformed["image"]

        img = F.to_tensor(img)
        annotation = {"image_id": torch.tensor([img_id]),
                      "boxes": torch.as_tensor(bboxes, dtype=torch.int16),
                      "labels": torch.as_tensor(labels, dtype=torch.int64),
                      "area": torch.as_tensor(areas, dtype=torch.float32),
                      "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
                      "keypoints": torch.as_tensor(keypoints, dtype=torch.float32).view(-1, self.points_num, 3)}

        return img, annotation

    def __len__(self):
        return len(self.ids)

    def augment(self, img, key_points, bboxes, labels, augment_val=False):
        if not augment_val:
            one_of = [alb.ImageCompression(p=0.8),
                      alb.Blur(blur_limit=5, p=0.8),
                      alb.GaussNoise(p=0.8),
                      alb.CLAHE(p=0.8),
                      alb.RandomGamma(p=0.8),
                      ]

            transform = alb.Compose([
                # alb.Resize(height=640, width=640),
                # alb.RandomCrop(500, 500, always_apply=False, p=0.3),
                alb.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, fit_output=False, interpolation=1,
                                always_apply=False, p=0.8),
                alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=180, border_mode=0, value=0,
                                     always_apply=False, p=0.8),
                alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, always_apply=False, p=0.7),
                alb.ChannelShuffle(p=0.5),
                alb.RGBShift(p=0.5),
                alb.OneOf(one_of, p=0.8),
                # alb.Normalize(mean=self.dataset_mean,
                #               std=self.dataset_std,
                #               max_pixel_value=self.max_pixel_value)
                ],
                keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False),
                bbox_params=alb.BboxParams(format='pascal_voc', min_area=self.min_area, label_fields=['bboxes_labels'])
            )
        else:
            transform = alb.Compose([
                # alb.Resize(height=640, width=640),
                # alb.Normalize(mean=self.dataset_mean,
                #               std=self.dataset_std,
                #               max_pixel_value=self.max_pixel_value)
                ],
                keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False),
                bbox_params=alb.BboxParams(format='pascal_voc', min_area=self.min_area, label_fields=['bboxes_labels'])
            )
        return transform(image=img, keypoints=key_points, bboxes=bboxes, bboxes_labels=labels)


def collate_function(batch):
    return tuple(zip(*batch))


def load_data(images_dir, annotation_file="../coco-1659778596.546996.json",
              transform=True, transform_val=False, shuffle=False, val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO):
    dataset = KeypointsDataset(images_dir=images_dir, annotation_file=annotation_file, transforms=transform,
                               transforms_val=transform_val)
    total_cnt = dataset.__len__()
    val_cnt = int(val_ratio * total_cnt)
    test_cnt = int(test_ratio * total_cnt)
    train_cnt = total_cnt - val_cnt - test_cnt

    print(f"\nTRAINING IMAGES NUMBER: {train_cnt}\n")
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(dataset, (train_cnt, val_cnt, test_cnt))
    dataset_train.transforms = transform
    dataset_val.transforms_val = transform_val

    dataloaders = {
        name: DataLoader(dataset=dataset_, collate_fn=collate_function, batch_size=cfg.BATCH_SIZE,
                         shuffle=shuffle, num_workers=cfg.NUM_WORKERS)
        for name, dataset_ in zip(["train", "val", "test"], [dataset_train, dataset_val, dataset_test])
    }

    return dataloaders


def get_normalization_params(images_dir="../images/train", annotation_file="../coco-1659778596.546996.json"):
    dataset = KeypointsDataset(images_dir=images_dir, annotation_file=annotation_file, transforms=False)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=False)
    mean, std, imgs_num = 0, 0, 0

    for imgs, _ in dataloader:
        imgs_cnt = imgs.size(0)
        imgs = imgs.view(imgs_cnt, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        imgs_num += imgs_cnt

    mean /= imgs_num
    std /= imgs_num

    return {"mean": mean.tolist(), "std": std.tolist()}


def check_examples():
    random.seed(1)
    dataloader = load_data(images_dir="../images/train", transform=True, transform_val=False)["train"]

    for img, annotation in dataloader:
        img = (img[0].numpy() * 255).astype(np.uint8)
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        keypoints, bboxes = annotation[0]["keypoints"], annotation[0]["boxes"]

        for obj_keypoints, bbox in zip(keypoints, bboxes):
            pt1, pt2 = tuple(bbox[:2]), tuple(bbox[2:])
            img = cv2.rectangle(img.copy(), pt1, pt2, (255, 0, 0), 2)
            for keypoint in obj_keypoints:
                center = (int(round(keypoint[0].item())), int(round(keypoint[1].item())))
                img = cv2.circle(img.copy(), center, 2, (0, 0, 255), 5)

        print(f"IMG SHAPE: {img.shape}")
        cv2.imshow("img", cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1500)


if __name__ == '__main__':
    norm_params = get_normalization_params(annotation_file="../coco-1659778596.546996.json")
    print(f"MEAN:{norm_params['mean']}\nSTD:{norm_params['std']}\n")
    check_examples()
