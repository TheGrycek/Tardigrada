import random
from pathlib import Path

import albumentations as alb
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

import keypoints_detector.config as cfg
from keypoints_detector.utils import tensor2rgb, random_bbox_crop_roi

seed = 123
random.seed(123)


class KeypointsDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=False):
        super().__init__()
        self.dataset_dir = Path(images_dir)
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.points_num = cfg.KEYPOINTS
        self.min_area = 25
        self.min_visibility = 0.8

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        img_path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(str(self.dataset_dir / img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        objects_num = len(coco_annotation)
        bboxes, areas, keypoints, labels, iscrowd = [], [], [], [], []

        for i in range(objects_num):
            coco_elem = coco_annotation[i]
            x_min = coco_elem['bbox'][0]
            y_min = coco_elem['bbox'][1]
            x_max = x_min + coco_elem['bbox'][2]
            y_max = y_min + coco_elem['bbox'][3]

            bboxes.append([x_min, y_min, x_max, y_max])
            keypoints.append(np.asarray(coco_elem['keypoints']).reshape((self.points_num, 3))[:, :2])
            labels.append(coco_elem["category_id"])
            areas.append(coco_elem['area'])
            iscrowd.append(coco_elem["iscrowd"])

        if self.transforms:
            img, bboxes, keypoints = self.augment(img,
                                                  np.asarray(bboxes),
                                                  np.asarray(keypoints).reshape(-1, 2))
        else:
            keypoints = np.asarray(keypoints).reshape((-1, self.points_num, 2))
            visibility = np.ones((keypoints.shape[0], self.points_num, 1)) * 2
            keypoints = np.concatenate((keypoints, visibility), axis=2)

        annotation = {"image_id": torch.tensor([img_id]),
                      "boxes": torch.as_tensor(np.asarray(bboxes), dtype=torch.int16),
                      "labels": torch.as_tensor(np.asarray(labels), dtype=torch.int64),
                      "area": torch.as_tensor(np.asarray(areas), dtype=torch.float32),
                      "iscrowd": torch.as_tensor(np.asarray(iscrowd), dtype=torch.int64),
                      "keypoints": torch.as_tensor(keypoints, dtype=torch.float32)}

        return F.to_tensor(img), annotation

    def set_transforms(self, transforms):
        self.transforms = transforms

    def augment(self, img, bboxes, keypoints):
        bboxes_instance = [i for i in range(bboxes.shape[0])]
        kpts_instance = [i for i in range(int(keypoints.shape[0] / self.points_num)) for _ in range(self.points_num)]

        while True:
            one_of = [alb.ImageCompression(p=1),
                      alb.Blur(blur_limit=10, p=1),
                      alb.GaussNoise(p=1),
                      alb.CLAHE(p=1),
                      alb.FancyPCA(alpha=0.1, p=1),
                      alb.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1)
                      ]
            x_min, y_min, x_max, y_max = random_bbox_crop_roi(bboxes, img.shape)
            transforms = [
                alb.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, p=0.5),
                alb.SmallestMaxSize(max_size=800,
                                    interpolation=1),
                *[alb.RandomRotate90(p=0.3) for _ in range(3)],
                alb.Affine(interpolation=3,
                           rotate_method="ellipse",
                           p=0.9),
                alb.RandomShadow(shadow_roi=(0, 0, 1, 1),
                                 num_shadows_lower=2,
                                 num_shadows_upper=5,
                                 shadow_dimension=5,
                                 p=0.5),
                alb.RandomSunFlare(flare_roi=(0, 0, 1, 1),
                                   angle_lower=0,
                                   angle_upper=1,
                                   num_flare_circles_lower=2,
                                   num_flare_circles_upper=6,
                                   src_radius=200,
                                   p=0.5),
                alb.ColorJitter(brightness=(0.5, 1.2),
                                contrast=(0.2, 1.0),
                                saturation=(0.2, 1.0),
                                hue=(-0.5, 0.5),
                                p=1),
                *[alb.OneOf(one_of, p=0.5) for _ in range(3)]
            ]

            transform = alb.Compose(transforms,
                                    keypoint_params=alb.KeypointParams(format='xy',
                                                                       remove_invisible=False,
                                                                       label_fields=['kpts_instance']),
                                    bbox_params=alb.BboxParams(format='pascal_voc',
                                                               min_area=self.min_area,
                                                               min_visibility=self.min_visibility,
                                                               label_fields=['bboxes_instance']))
            transformed = transform(image=img,
                                    keypoints=keypoints,
                                    bboxes=bboxes,
                                    bboxes_instance=bboxes_instance,
                                    kpts_instance=kpts_instance)

            keypoints_ = np.asarray(transformed["keypoints"]).reshape((-1, self.points_num, 2))
            visibility_ = np.ones((keypoints_.shape[0], self.points_num, 1)) * 2
            keypoints_ = np.concatenate((keypoints_, visibility_), axis=2)
            keypoints_ = keypoints_[transformed['bboxes_instance']]

            if len(transformed['bboxes']) > 0:
                return transformed["image"], transformed['bboxes'], keypoints_


def collate_function(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
        images_dir=cfg.IMAGES_PATH,
        annotation_file=cfg.ANNOTATION_FILE_PATH,
        transform_train=True,
        transform_val=False,
        transform_test=False,
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        batch_size=cfg.BATCH_SIZE, tile=False
):

    dataset = KeypointsDataset(images_dir=images_dir, annotation_file=annotation_file)
    total_cnt = dataset.__len__()
    val_cnt = int(val_ratio * total_cnt)
    test_cnt = int(test_ratio * total_cnt)
    train_cnt = total_cnt - val_cnt - test_cnt
    generator = torch.Generator().manual_seed(seed)

    print(f"\nTRAINING IMAGES NUMBER: {train_cnt}\n"
          f"VALIDATION IMAGES NUMBER: {val_cnt}\n"
          f"TESTING IMAGES NUMBER: {test_cnt}\n")

    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset,
        (train_cnt, val_cnt, test_cnt),
        generator=generator
    )
    datasets = {
        "train": (dataset_train, transform_train, shuffle_train),
        "val": (dataset_val, transform_val, shuffle_val),
        "test": (dataset_test, transform_test, shuffle_test)
    }

    dataloaders = {}
    for name, (subset, transform, shuffle) in datasets.items():
        if len(subset) > 0:
            # TODO: refactor to not override dataset classes
            subset.dataset = KeypointsDataset(images_dir=images_dir,
                                              annotation_file=annotation_file,
                                              transforms=transform)
            dataloaders[name] = DataLoader(dataset=subset,
                                           collate_fn=collate_function,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=cfg.NUM_WORKERS,
                                           worker_init_fn=seed_worker,
                                           generator=generator)
        else:
            dataloaders[name] = None

    return dataloaders


def get_normalization_params(images_dir=cfg.IMAGES_PATH, annotation_file=cfg.ANNOTATION_FILE_PATH):
    dataloader = create_dataloaders(images_dir=images_dir,
                                    annotation_file=annotation_file,
                                    transform_train=False,
                                    shuffle_train=False,
                                    val_ratio=cfg.VAL_RATIO,
                                    test_ratio=cfg.TEST_RATIO,
                                    batch_size=1)
    mean, std, imgs_num = 0, 0, 0

    for imgs, _ in dataloader["train"]:
        imgs = torch.stack(imgs)
        imgs_cnt = imgs.size(0)
        imgs = imgs.view(imgs_cnt, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        imgs_num += imgs_cnt

    mean /= imgs_num
    std /= imgs_num

    norm_params = {"mean": mean.tolist(), "std": std.tolist()}
    print(f"MEAN:{norm_params['mean']}\nSTD:{norm_params['std']}\n")

    return norm_params


def check_examples():
    dataloader = create_dataloaders(images_dir=cfg.IMAGES_PATH, transform_train=True)["train"]

    for i, (img, annotation) in enumerate(dataloader):
        img = tensor2rgb(img)
        keypoints, bboxes = annotation[0]["keypoints"], annotation[0]["boxes"]

        for obj_keypoints, bbox in zip(keypoints, bboxes):
            pt1, pt2 = tuple(bbox[:2]), tuple(bbox[2:])
            img = cv2.rectangle(img.copy(), pt1, pt2, (255, 0, 0), 2)
            for keypoint in obj_keypoints:
                center = (int(keypoint[0]), int(keypoint[1]))
                img = cv2.circle(img.copy(), center, 4, (0, 100, 255), 5)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow("img", img)
        cv2.waitKey(1500)


if __name__ == '__main__':
    get_normalization_params()
    check_examples()
