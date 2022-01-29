import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import keypoints_detector.config as cfg
from keypoints_detector.model import keypoint_detector
from keypoints_detector.predict import predict
from scale_detector.scale_detector import read_scale


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, default="./images/",
                        help="Input images directory.")

    return parser.parse_args()


def filter_cnts(cnts):
    cnts_filtered = []
    bboxes_fit = []
    bboxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 1000:
            continue
        rect_tilted = cv2.minAreaRect(cnt)
        if 0.5 < rect_tilted[1][0] / rect_tilted[1][1] < 2:
            continue
        rect = cv2.boundingRect(cnt)
        bboxes.append(rect)
        cnts_filtered.append(cnt)
        bboxes_fit.append(rect_tilted)

    return cnts_filtered, bboxes_fit, bboxes


def simple_segmenter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts, bboxes_fit, bboxes = filter_cnts(cnts)

    return cnts, bboxes_fit, bboxes, thresh


def calculate_mass(predicted, scale):
    scale_ratio = scale["um"] / scale["pix"]
    masses = np.array([])

    for i, points in enumerate(predicted["keypoints"]):
        points = points.astype(np.uint64)
        head, ass, right, left = points
        length = np.sqrt(np.sum((head - ass) ** 2)) * scale_ratio
        width = np.sqrt(np.sum((length - right) ** 2)) * scale_ratio

        class_name = cfg.INSTANCE_CATEGORY_NAMES[predicted["labels"][i]]
        mass = 0.0
        R = 0.0

        if length and width != np.nan:
            R = length / width
            density = 1.04

            # If the mass of the species Echiniscus is estimated, use different equation
            if class_name == 'echiniscus':
                mass = (1 / 12) * length * np.pi * (length / R) ** 0.5 * density * 10 ** -6  # [ug]
            else:
                mass = length * np.pi * (length / (2 * R)) ** 0.5 * density * 10 ** -6  # [ug]

        # print(f"length / width: {R}, mass: {mass}")
        masses = np.append(masses, mass)

    total_mass = np.sum(masses)
    mass_std = np.std(masses)
    mass_mean = np.mean(masses)

    return total_mass, mass_std, mass_mean


def main(args):
    images = args.input_dir.glob("*.jpg")

    model = keypoint_detector()
    model.load_state_dict(torch.load("keypoints_detector/checkpoints/keypoints_detector.pth"))

    for img_path in images:
        print(f"image path: {img_path}")

        img = cv2.imread(str(img_path), 1)
        cnts, bboxes_fit, bboxes, thresh = simple_segmenter(img)

        image_scale, img = read_scale(img, bboxes[0], device="cpu")
        print(f"Image scale: {image_scale}")

        predicted = predict(model, img, device=cfg.DEVICE)
        biomass, biomass_std, biomass_mean = calculate_mass(predicted, scale=image_scale)

        img = predicted["image"]

        print(f"Image total mass: {biomass} ug\n")
        print("-" * 50)

        info_dict = {"scale": (f"Scale: {image_scale['um']} um", (50, 50)),
                     "number": (f"Animal number: {predicted['bboxes'].shape[0]}", (50, 100)),
                     "mass": (f"Total biomass: {round(biomass, 5)} ug", (50, 150)),
                     "mean": (f"Animal mass mean: {round(biomass_mean, 5)} ug", (50, 200)),
                     "std": (f"Biomass std: {round(biomass_std, 5)} ug", (50, 250))}

        for text, position in info_dict.values():
            img = cv2.putText(img, text, position,
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imwrite("../images/keypoint_rcnn_detection.jpg", img)
        cv2.imshow('predicted', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1000)


if __name__ == '__main__':
    main(parse_args())
