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
from augmentations import scale_rotate, tensor2rgb
from config import KEYPOINTS

seed = 123
random.seed(123)


class KeypointsDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=True):
        super().__init__()
        self.dataset_dir = Path(images_dir)
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.points_num = KEYPOINTS

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

        keypoints = np.asarray(keypoints).reshape((-1, self.points_num, 3))
        if self.transforms:
            img, bboxes, keypoints, labels = self.augment(img, keypoints=keypoints, labels=labels, bboxes=bboxes)

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

    def augment(self, img, keypoints, bboxes, labels):
        # custom scale and rotation function- albumentations bug
            img, keypoints, bboxes, labels = scale_rotate(img, keypoints, bboxes, labels, keypoints_num=self.points_num)
            one_of = [alb.ImageCompression(p=1),
                      alb.Blur(blur_limit=5, p=1),
                      alb.GaussNoise(p=1),
                      alb.CLAHE(p=1),
                      alb.RandomGamma(p=1),
                      alb.Downscale(scale_min=0.25, scale_max=0.75, p=1),
                      alb.FancyPCA(alpha=0.1, p=1),
                      ]

            transform = alb.Compose([
                *[alb.OneOf(one_of, p=0.6) for _ in range(3)],
                alb.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.7),
                alb.ChannelShuffle(p=0.5),
                alb.RGBShift(p=0.5),
                ],
                keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False),
                bbox_params=alb.BboxParams(format='pascal_voc', min_area=self.min_area, label_fields=['bboxes_labels'])
            )
            transformed = transform(image=img, keypoints=keypoints, bboxes=bboxes, bboxes_labels=labels)

            return transformed["image"], np.asarray(transformed['bboxes']), keypoints, labels


def collate_function(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data(images_dir, annotation_file=cfg.ANNOTATON_FILE,
              transform=True, shuffle=False, val_ratio=cfg.VAL_RATIO, test_ratio=cfg.TEST_RATIO):
    dataset = KeypointsDataset(images_dir=images_dir, annotation_file=annotation_file, transforms=transform)
    total_cnt = dataset.__len__()
    val_cnt = int(val_ratio * total_cnt)
    test_cnt = int(test_ratio * total_cnt)
    train_cnt = total_cnt - val_cnt - test_cnt
    generator = torch.Generator().manual_seed(1)

    print(f"\nTRAINING IMAGES NUMBER: {train_cnt}\n"
          f"VALIDATION IMAGES NUMBER: {val_cnt}\n"
          f"TESTING IMAGES NUMBER: {test_cnt}\n")
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset,
        (train_cnt, val_cnt, test_cnt),
        generator=generator
    )
    dataset_train.transforms = transform

    dataloaders = {
        name: DataLoader(dataset=dataset_, collate_fn=collate_function, batch_size=cfg.BATCH_SIZE,
                         shuffle=shuffle, num_workers=cfg.NUM_WORKERS, worker_init_fn=seed_worker, generator=generator)
        for name, dataset_ in zip(["train", "val", "test"], [dataset_train, dataset_val, dataset_test])
    }

    return dataloaders


def get_normalization_params(images_dir="../images/train", annotation_file=cfg.ANNOTATON_FILE):
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
    dataloader = load_data(images_dir="../images/train", transform=True)["val"]

    for i, (img, annotation) in enumerate(dataloader):
        img = tensor2rgb(img)
        keypoints, bboxes = annotation[0]["keypoints"], annotation[0]["boxes"]

        for obj_keypoints, bbox in zip(keypoints, bboxes):
            pt1, pt2 = tuple(bbox[:2]), tuple(bbox[2:])
            img = cv2.rectangle(img.copy(), pt1, pt2, (255, 0, 0), 2)
            for keypoint in obj_keypoints:
                center = (int(round(keypoint[0].item())), int(round(keypoint[1].item())))
                img = cv2.circle(img.copy(), center, 2, (0, 0, 255), 5)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, None, fx=0.7, fy=0.7)
        cv2.imshow("img", cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1500)


if __name__ == '__main__':
    # norm_params = get_normalization_params(annotation_file="../coco-1659778596.546996.json")
    # print(f"MEAN:{norm_params['mean']}\nSTD:{norm_params['std']}\n")
    check_examples()
