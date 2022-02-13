import argparse
import json
import logging
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--json_path", default="./coco-jakub.json", type=Path,
                        help="Path to the json annotation file")
    parser.add_argument("-n", "--keypoints_num", default=7, type=int,
                        help="Number of keypoints for an instance")
    return parser.parse_args()


def print_info(info_type, annotation, images, categories, keypoints_num=None, warn_pts_num=None):
    info = f"IMAGE NAME: {images[annotation['image_id']]}\n" \
           f"INSTANCE ID: {annotation['id']}\n" \
           f"CATEGORY: {categories[annotation['category_id']]}\n" \
           f"{'-' * 30}"

    if info_type == "error":
        info_err = f"\nWRONG KEYPOINTS NUMBER!" \
                   f"KEYPOINTS NUMBER: {keypoints_num}\n"

        logging.error(info_err + info)

    elif info_type == "warning":
        info_war = f"\nKEYPOINTS OUTSIDE THE BOUNDING BOX!\n" \
                   f"WRONG KEYPOINTS NUMBER: {warn_pts_num}\n" \

        logging.warning(info_war + info)


def check_box_points(bbox, points, points_num):
    pts = np.asarray(points).reshape(points_num, 3)[:, :2]

    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
    wrong_pts_num = 0

    for x, y in pts:
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            wrong_pts_num += 1

    return wrong_pts_num


def check_json(args):
    ann_path = args.json_path
    keypoints_num = args.keypoints_num
    errors_num = 0
    warnings_num = 0

    with open(str(ann_path), "r+") as f:
        ann_file = json.load(f)

    images = {img["id"]: img["file_name"] for img in ann_file["images"]}
    categories = {cat["id"]: cat["name"] for cat in ann_file["categories"]}

    for annotation in ann_file["annotations"]:
        if "num_keypoints" not in annotation.keys():
            k_num = 0
            print_info("error", annotation, images, categories, keypoints_num=k_num)
            errors_num += 1

        elif annotation["num_keypoints"] != keypoints_num:
            k_num = annotation['num_keypoints']
            print_info("error", annotation, images, categories, keypoints_num=k_num)
            errors_num += 1

        else:
            wrong_pts_num = check_box_points(annotation["bbox"], annotation["keypoints"], keypoints_num)
            if wrong_pts_num > 0:
                print_info("warning", annotation, images, categories, warn_pts_num=wrong_pts_num)
                warnings_num += 1

    logging.info(f"TOTAL NUMBER OF ERRORS: {errors_num}")
    logging.info(f"TOTAL NUMBER OF WARNINGS: {warnings_num}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    check_json(parse_args())
