import argparse
import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from utils import COCOFormat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, default="./images/krio5_OM_1.5_1.jpg",
                        help="Input images directory.")
    parser.add_argument("-o", "--output_dir", type=Path, default="./images/",
                        help="Output images directory.")

    return parser.parse_args()


def callback(x):
    global low_h, high_h, low_s, high_s, low_v, high_v, show_ann, min_lr, max_lr, min_area, max_area, white_border_pix

    low_h = cv2.getTrackbarPos('low H', 'control_panel')
    high_h = cv2.getTrackbarPos('high H', 'control_panel')
    low_s = cv2.getTrackbarPos('low S', 'control_panel')
    high_s = cv2.getTrackbarPos('high S', 'control_panel')
    low_v = cv2.getTrackbarPos('low V', 'control_panel')
    high_v = cv2.getTrackbarPos('high V', 'control_panel')

    show_ann = cv2.getTrackbarPos('show_annotation', 'control_panel')
    min_lr = cv2.getTrackbarPos('min length/width ratio', 'control_panel')
    max_lr = cv2.getTrackbarPos('max length/width ratio', 'control_panel')
    min_area = cv2.getTrackbarPos('min area', 'control_panel')
    max_area = cv2.getTrackbarPos('max area', 'control_panel')
    white_border_pix = cv2.getTrackbarPos('white border', 'control_panel')


def filter_cnts(cnts):
    global min_lr, max_lr, min_area, max_area

    cnts_filtered = []
    areas = []
    bboxes = []
    for cnt in cnts:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > 0:
            if max_area >= cnt_area >= min_area:
                rect_tilted = cv2.minAreaRect(cnt)
                if min_lr <= rect_tilted[1][0] / rect_tilted[1][1] <= max_lr:
                    rect = cv2.boundingRect(cnt)
                    cnts_filtered.append(cnt)
                    areas.append(cnt_area)
                    bboxes.append(rect)
    return cnts_filtered, areas, bboxes


def find_contours(mask):
    thresh = cv2.bitwise_not(mask)
    cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts, areas, bboxes = filter_cnts(cnts)
    return cnts, areas, bboxes


def white_borders(mask):
    global white_border_pix

    if white_border_pix > 0:
        h, w = mask.shape
        mask[0: white_border_pix, :] = 255
        mask[:, 0:white_border_pix] = 255
        mask[-white_border_pix: h, :] = 255
        mask[:, -white_border_pix: w] = 255

    return mask


def save_annotations(cnts, areas, bboxes, img_name, img_shape, img_num=1):
    current_date = datetime.datetime.now()
    current_date = str(current_date)

    for char in (" ", ":", ".", "-"):
        current_date = current_date.replace(char, "-")

    file_name = "labels_tardigrada_" + current_date + ".json"

    coco = COCOFormat()
    coco_format = coco.coco_format

    image_dict = coco.image
    image_dict["id"] = img_num
    image_dict["width"] = img_shape[1]
    image_dict["height"] = img_shape[0]
    image_dict["file_name"] = img_name

    coco_format["images"].append(image_dict)

    new_cnts = [cnt.flatten().tolist() for cnt in cnts]

    for i, (cnt, area, bbox) in enumerate(zip(new_cnts, areas, bboxes)):
        annotation_dict = coco.annotations.copy()
        annotation_dict["id"] = i
        annotation_dict["category_id"] = 2
        annotation_dict["image_id"] = img_num
        annotation_dict["area"] = area
        annotation_dict["segmentation"] = [cnt]
        annotation_dict["bbox"] = bbox

        coco_format["annotations"].append(annotation_dict)

    out_file = open(file_name, "w")
    json.dump(coco_format, out_file, indent=6)
    out_file.close()


def main(args):
    img_path = args.input_dir
    img = cv2.imread(str(img_path))
    img_shape = img.shape

    cv2.namedWindow('control_panel', 2)
    cv2.resizeWindow("control_panel", 550, 300)

    cv2.createTrackbar('low H', 'control_panel', 0, 179, callback)
    cv2.createTrackbar('high H', 'control_panel', 179, 179, callback)

    cv2.createTrackbar('low S', 'control_panel', 0, 255, callback)
    cv2.createTrackbar('high S', 'control_panel', 255, 255, callback)

    cv2.createTrackbar('low V', 'control_panel', 170, 255, callback)
    cv2.createTrackbar('high V', 'control_panel', 255, 255, callback)

    cv2.createTrackbar('show_annotation', 'control_panel', 0, 1, callback)

    cv2.createTrackbar('max length/width ratio', 'control_panel', 200, 200, callback)
    cv2.createTrackbar('min length/width ratio', 'control_panel', 0, 200, callback)

    cv2.createTrackbar('min area', 'control_panel', 800, 30000, callback)
    cv2.createTrackbar('max area', 'control_panel', 50000, 50000, callback)

    cv2.createTrackbar('white border', 'control_panel', 120, 300, callback)

    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_low = np.array([low_h, low_s, low_v], np.uint8)
        hsv_high = np.array([high_h, high_s, high_v], np.uint8)

        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        res = cv2.bitwise_and(img, img, mask=mask)

        mask = white_borders(mask)
        cnts, areas, bboxes = find_contours(mask)

        if show_ann == 1:
            cv2.drawContours(res, cnts, -1, (0, 0, 255), 3)

        cv2.imshow('mask', cv2.resize(mask, (1200, 800)))
        cv2.imshow('masked_img', cv2.resize(res, (1200, 800)))

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("Keyboard Interrupt.")
            save_annotations(cnts, areas, bboxes, img_path.name, img_shape)
            break
        elif k == 115:
            save_annotations(cnts, areas, bboxes, img_path.name, img_shape)
            print("Annotations saved.")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    low_h, low_s, low_v = 0, 0, 0
    high_h, high_s, high_v = 179, 255, 255
    min_area, max_area = 0, 300000
    min_lr, max_lr = 1, 1000
    show_ann = 1
    white_border_pix = 0

    main(parse_args())
