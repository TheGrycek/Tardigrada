from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

import config as cfg


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
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

        bboxes, areas, masks, labels, iscrowd = [], [], [], [], []
        for i in range(objects_num):
            coco_elem = coco_annotation[i]
            x_min = coco_elem['bbox'][0]
            y_min = coco_elem['bbox'][1]
            x_max = x_min + coco_elem['bbox'][2]
            y_max = y_min + coco_elem['bbox'][3]

            bboxes.append([x_min, y_min, x_max, y_max])
            areas.append(coco_elem['area'])
            mask = coco.annToMask(coco_elem)
            masks.append(mask)

            labels.append(coco_elem["category_id"])
            iscrowd.append(coco_elem["iscrowd"])

        annotation = {"image_id": torch.tensor([img_id]),
                      "boxes": torch.as_tensor(bboxes, dtype=torch.int16),
                      "labels": torch.as_tensor(labels, dtype=torch.int64),
                      "area": torch.as_tensor(areas, dtype=torch.float32),
                      "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
                      "masks": torch.as_tensor(np.array(masks), dtype=torch.float32)}

        if self.transforms is not None:
            img = self.transforms(img)

        img_shape = img.shape
        img = torch.from_numpy(img.astype(np.float32)) / 255
        img = img.view([3, img_shape[0], img_shape[1]])

        return img, annotation

    def __len__(self):
        return len(self.ids)


def collate_function(batch):
    images_list = []
    targets_list = []

    for img, targets in batch:
        images_list.append(img)
        targets_list.append(targets)

    return images_list, targets_list


def load_data(images_dir, annotation_file="../labels_tardigrada_2022-01-07-20-11-35-797274.json",
              transform=True, shuffle=False):
    dataset = SegmentationDataset(images_dir=images_dir, annotation_file=annotation_file)
    dataloader = DataLoader(dataset=dataset, collate_fn=collate_function,
                            batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.NUM_WORKERS)

    return dataloader


if __name__ == '__main__':
    dataloader = load_data(images_dir="../images")

    for img, annotation in dataloader:
        print(annotation)
