import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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


def calculate_mass(predicted, scale, img_path):
    scale_ratio = scale["um"] / scale["pix"]
    density = 1.04
    results = []

    for i, points in enumerate(predicted["keypoints"]):
        points = points.astype(np.uint64)
        head, ass, right, left = points
        length = np.sqrt(np.sum((head - ass) ** 2)) * scale_ratio
        width = np.sqrt(np.sum((left - right) ** 2)) * scale_ratio

        class_name = cfg.INSTANCE_CATEGORY_NAMES[predicted["labels"][i]]

        if length and width != np.nan:
            R = length / width

            # If the mass of the species Echiniscus is estimated, use different equation
            if class_name == 'echiniscus':
                mass = (1 / 12) * length * np.pi * (length / R) ** 2 * density * 10 ** -6  # [ug]
            else:
                mass = length * np.pi * (length / (2 * R)) ** 2 * density * 10 ** -6  # [ug]

            # print(f"length: {length}, width: {width} R: {R}, mass: {mass}")
            results.append({"img_path": img_path,
                            "id": i,
                            "class": class_name,
                            "length": length,
                            "width": width,
                            "biomass": mass})

    return pd.DataFrame(results)


def main(args):
    Path("../results").mkdir(exist_ok=True, parents=True)
    # images = args.input_dir.glob("*.jpg")
    images = [Path("./images/krio5_OM_1.5_5.jpg")]
    model = keypoint_detector()
    model.load_state_dict(torch.load("keypoints_detector/checkpoints/keypoints_detector.pth"))

    for img_path in images:
        img = cv2.imread(str(img_path), 1)
        cnts, bboxes_fit, bboxes, thresh = simple_segmenter(img)
        image_scale, img = read_scale(img, bboxes[0], device="cpu")

        predicted = predict(model, img, device=cfg.DEVICE)
        results_df = calculate_mass(predicted, image_scale, img_path)

        mass_total = results_df["biomass"].sum()
        mass_mean = results_df["biomass"].mean()
        mass_std = results_df["biomass"].std()
        img = predicted["image"]

        print(f"image path: {img_path}\n"
              f"Image scale: {image_scale}\n"
              f"Image total mass: {mass_total} ug")
        print("-" * 50)

        info_dict = {"scale": (f"Scale: {image_scale['um']} um", (50, 50)),
                     "number": (f"Animal number: {predicted['bboxes'].shape[0]}", (50, 100)),
                     "mass": (f"Total biomass: {round(mass_total, 5)} ug", (50, 150)),
                     "mean": (f"Animal mass mean: {round(mass_mean, 5)} ug", (50, 200)),
                     "std": (f"Biomass std: {round(mass_std, 5)} ug", (50, 250))}

        for text, position in info_dict.values():
            img = cv2.putText(img, text, position,
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        results_df.to_csv(f"../results/{img_path.stem}_results.csv")
        cv2.imwrite("../images/keypoint_rcnn_detection.jpg", img)
        cv2.imshow('predicted', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(2500)


if __name__ == '__main__':
    main(parse_args())
