import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from key_points_detector.config import INPUT_IMAGE_SIZE
from key_points_detector.model import AlexNet
from scale_detector.scale_detector import read_scale
from segmenter.model import simple_segmenter
from utils import prepare_segmented_img, prepare_contours, resize_pad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, default="./images/",
                        help="Input images directory.")

    return parser.parse_args()


def calculate_mass(points, scale, img_ratio=1, echiniscus=False):
    # If the mass of the species Echiniscus is estimated, use different equation
    # TODO: pass nn input image scale factors and multiply with length and width
    head, ass, right, left = points
    length_pix = np.linalg.norm(head - ass) * img_ratio
    width_pix = np.linalg.norm(right - left) * img_ratio
    scale_ratio = scale["pix"] * scale["um"]
    length = scale_ratio / length_pix
    width = scale_ratio / width_pix

    R = length / width
    density = 1.04

    if echiniscus:
        mass = (1 / 12) * length * np.pi * (length / R) ** 0.5 * density * 10 ** -6  # [ug]
    else:
        mass = length * np.pi * (length / (2 * R)) ** 0.5 * density * 10 ** -6  # [ug]

    print(f"length / width: {R}, mass: {mass}")
    return mass


def main(args):
    images = args.input_dir.glob("*.jpg")

    model = AlexNet()
    model.load_state_dict(torch.load("./key_points_detector/checkpoints/pose_net.pth"))

    for img_path in images:
        print(f"image path: {img_path}")
        img = cv2.imread(str(img_path), 1)

        cnts, bboxes_fit, bboxes, thresh = simple_segmenter(img)
        first = True

        image_biomass = 0
        image_scale = {'pix': 1, 'um': 1}

        for c, rect_tilted, rect in zip(cnts, bboxes_fit, bboxes):
            if first:
                image_scale = read_scale(img, rect, device="cpu")
                print(f"Image scale: {image_scale}")
                first = False
                continue

            cnt_normalized, cnt_translated, center, shape_translated = prepare_contours(rect_tilted, rect, c)
            label_image = prepare_segmented_img(img, cnt_translated, shape_translated, rect)
            ratio, label_image = resize_pad(label_image, img_size=None, new_size=INPUT_IMAGE_SIZE)
            input_tensor = torch.from_numpy(label_image.astype(np.float32))
            input_tensor = input_tensor.view([1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE])

            with torch.no_grad():
                predicted = model(input_tensor).cpu()
                predicted = predicted.view([4, 2])

                mass = calculate_mass(predicted.numpy(), img_ratio=INPUT_IMAGE_SIZE / ratio, scale=image_scale)
                image_biomass += mass

        print(f"Image total mass: {image_biomass} ug\n")
        cv2.imshow('segmented', cv2.resize(thresh, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1000)


if __name__ == '__main__':
    main(parse_args())
