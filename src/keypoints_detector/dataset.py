import random
from pathlib import Path

import albumentations as alb
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

import config as cfg


class KeypointsDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=True):
        super().__init__()
        self.dataset_dir = Path(images_dir)
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

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
            keypoints = np.asarray(keypoints).reshape((-1, 4, 3))
            visibility = keypoints[:, :, 2].reshape((-1, 4, 1))
            keypoints_no_vis = keypoints[:, :, :2].reshape((-1, 2))
            transformed = self.augment(img, key_points=keypoints_no_vis, labels=labels, bboxes=bboxes)
            keypoints = np.asarray(transformed["keypoints"]).reshape((-1, 4, 2))
            keypoints = np.concatenate((keypoints, visibility), axis=2)
            bboxes = np.asarray(transformed['bboxes'])
            img = transformed["image"]

        img = F.to_tensor(img)
        annotation = {"image_id": torch.tensor([img_id]),
                      "boxes": torch.as_tensor(bboxes, dtype=torch.int16),
                      "labels": torch.as_tensor(labels, dtype=torch.int64),
                      "area": torch.as_tensor(areas, dtype=torch.float32),
                      "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
                      "keypoints": torch.as_tensor(keypoints, dtype=torch.float32).view(-1, 4, 3)}

        return img, annotation

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def augment(img, key_points, bboxes, labels):

        one_of = [alb.ImageCompression(p=0.8),
                  alb.Blur(blur_limit=5, p=0.8),
                  alb.GaussNoise(p=0.8),
                  alb.CLAHE(p=0.8),
                  alb.RandomGamma(p=0.8),
                  ]

        one_of2 = [alb.HueSaturationValue(p=0.5),
                   alb.RGBShift(p=0.7)]

        transform = alb.Compose([
            alb.Affine(p=0.9),
            alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            alb.OneOf(one_of, p=0.8),
            alb.OneOf(one_of, p=0.8),
            alb.OneOf(one_of2, p=0.5),
            # alb.normalize(mean=[], std=[], max_pixel_value=255)
            ],
            keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False),
            bbox_params=alb.BboxParams(format='pascal_voc', min_area=100, label_fields=['bboxes_labels'])
        )

        return transform(image=img, keypoints=key_points, bboxes=bboxes, bboxes_labels=labels)


def collate_function(batch):
    return tuple(zip(*batch))


def load_data(images_dir, annotation_file="../Tardigrada-5.json",
              transform=False, shuffle=False):
    dataset = KeypointsDataset(images_dir=images_dir, annotation_file=annotation_file, transforms=transform)
    dataloader = DataLoader(dataset=dataset, collate_fn=collate_function,
                            batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.NUM_WORKERS)

    return dataloader


if __name__ == '__main__':
    random.seed(1)
    dataloader = load_data(images_dir="../images")

    for i in range(100):
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
                    img = cv2.circle(img.copy(), center, 5, (0, 0, 255), 5)

            cv2.imshow("img", cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(1500)
