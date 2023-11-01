#!/usr/bin/env python3
import argparse
import logging
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import ujson
from scipy.interpolate import interp1d

import keypoints_detector.config as cfg
from keypoints_detector.model import KeypointDetector
from keypoints_detector.predict import predict
from keypoints_detector.utils import calc_dimensions
from scale_detector.scale_detector import read_scale


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, default="./keypoints_detector/datasets/test",
                        help="Input images directory.")
    parser.add_argument("-o", "--output_dir", type=Path, default="../results/",
                        help="Outputs directory.")

    return parser.parse_args()


def fit_spline(bbox, points, keypoints_num, method="quadratic", num_pts_inter=30):
    try:
        length_pts = points[: keypoints_num - 2]
        distance = np.cumsum(np.sqrt(np.sum(np.diff(length_pts, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]
        interpolator = interp1d(distance, length_pts, kind=method, axis=0)
        pts = interpolator(np.linspace(0, 1, num_pts_inter))

    except:
        print("\nBspline error! No points correction added!")
        print(traceback.format_exc())
        pts = np.reshape(np.array(bbox), ((bbox.size // 2), 2))
        return pts

    return pts


def calculate_mass(predicted, scale, img_path, curve_fit_algorithm="quadratic", queue=None):
    """'Curve fit algorithms: [slinear', 'quadratic', 'cubic]'"""

    if len(scale["bbox"]) == 0 or scale["bbox"][2] == 0:
        error = f"[Error!] Empty scale bbox or scale length equals zero for image {img_path}"
        logging.error(error)
        if queue is not None:
            queue.put(str(error) + "\n")
        return pd.DataFrame(columns=["img_path", "id", "class", "length", "width", "biomass"]), []

    scale_ratio = scale["um"] / scale["bbox"][2]
    density = 1.04
    results = []
    lengths_points = []

    for i, (bbox, points) in enumerate(zip(predicted["bboxes"], predicted["keypoints"])):
        length_pts = fit_spline(bbox, points, cfg.KEYPOINTS, method=curve_fit_algorithm)
        lengths_points.append(length_pts)
        length, width = calc_dimensions(length_pts, points[-2:], scale_ratio)
        class_name = cfg.INSTANCE_CATEGORY_NAMES[predicted["labels"][i]]

        if length and width != np.nan:
            R = length / width

            # If the mass of the Echiniscus species is estimated, use different equation
            if class_name == 'heter_ech':
                mass = (1 / 12) * length * np.pi * (length / R) ** 2 * density * 10 ** -6  # [ug]
            else:
                mass = length * np.pi * (length / (2 * R)) ** 2 * density * 10 ** -6  # [ug]

            info = {
                "img_path": img_path,
                "id": i,
                "class": class_name,
                "length": length,
                "width": width,
                "biomass": mass
            }

            results.append(info)

    return pd.DataFrame(results), lengths_points


def prepare_paths(args):
    args.output_dir.mkdir(exist_ok=True, parents=True)
    images_extensions = ("png", "tif", "jpg", "jpeg")

    images_paths = []
    for ext in images_extensions:
        ext_paths = list(args.input_dir.glob(f"*.{ext}"))
        images_paths.extend(ext_paths)

    return images_paths


def log_error_and_queue(queue, info):
    logging.error(info)
    queue.put(str(info) + "\n")


def run_inference(args, queue, stop, image_scale=None):
    images_paths = prepare_paths(args)
    model = KeypointDetector(tiling=True)

    for i, img_path in enumerate(images_paths, start=1):
        if not Path(img_path).is_dir():
            log_error_and_queue(queue, "Images directory does not exist!")

        if len(list(Path(img_path).glob("*"))) == 0:
            log_error_and_queue(queue, "Empty images directory!")

        if stop():
            queue.put("Processing stopped.\n")
            break

        try:
            img = cv2.imread(str(img_path), 1)

            if image_scale is None:
                image_scale, _ = read_scale(img, device="cpu")

            predicted = predict(model, img)
            out_dict = {
                "path": str(img_path),
                "scale_bbox": image_scale["bbox"],
                "scale_value": image_scale["um"],
                "annotations": []
            }

            for j in range(len(predicted["labels"])):
                result_dict = {
                    "label": int(predicted["labels"][j]),
                    "bbox": predicted["bboxes"][j].tolist(),
                    "keypoints": predicted["keypoints"][j].tolist(),
                    "score": float(predicted["scores"][j])
                }
                out_dict["annotations"].append(result_dict)

            out_path = args.output_dir / (img_path.stem + ".json")
            ujson.dump(out_dict, out_path.open("w"))
            queue.put(f"[{i}/{len(images_paths)}] Image inferenced: {str(img_path)}\n")

        except:
            log_error_and_queue(queue, traceback.format_exc())

    if not stop():
        queue.put(f"Inference finished.\n")


def run_calc_mass(args, queue, stop, curve_fit_algorithm="quadratic"):
    json_paths = list(args.output_dir.glob(f"*.json"))

    for i, json_path in enumerate(json_paths, start=1):
        if stop():
            queue.put("Processing stopped.\n")
            break

        annotation_dict = ujson.load(json_path.open("r"))
        img_path = Path(annotation_dict["path"])
        image_scale = {
            "um": annotation_dict["scale_value"],
            "bbox": annotation_dict["scale_bbox"]
        }
        predicted = {"keypoints": [], "bboxes": [], "labels": []}

        for annot in annotation_dict["annotations"]:
            predicted["keypoints"].append(annot["keypoints"])
            predicted["bboxes"].append(annot["bbox"])
            predicted["labels"].append(annot["label"])

        predicted["keypoints"] = np.round(predicted["keypoints"], 0)
        results_df, lengths_points = calculate_mass(predicted, image_scale, str(img_path),
                                                    curve_fit_algorithm, queue)
        results_df.to_csv(args.output_dir / f"{img_path.stem}_results.csv")

        queue.put(f"[{i}/{len(json_paths)}] Mass calculated: {str(img_path)}\n")

    if not stop():
        queue.put(f"Mass calculation finished.\n")


def visualize(args):
    images_paths = prepare_paths(args)
    model = KeypointDetector()

    for img_path in images_paths:
        try:
            start = time.time()
            img = cv2.imread(str(img_path), 1)
            image_scale, img = read_scale(img, device="cpu", visualize=True)
            predicted = predict(model, img)
            results_df, lengths_points = calculate_mass(predicted, image_scale, img_path)

            img = predicted["image"]

            if not results_df.empty:
                mass_total = results_df["biomass"].sum()
                mass_mean = results_df["biomass"].mean()
                mass_std = results_df["biomass"].std()

                print(f"image path: {img_path}\n"
                      f"Image scale: {image_scale}\n"
                      f"Image total mass: {mass_total} ug\n"
                      f"{'-' * 50}")
                print(F"Image processed: {str(img_path)}\n")
                shift, dec = 50, 5  # shift the text from the border, round to 5 decimal places
                info_dict = {"scale": (f"Scale: {image_scale['um']} um", (shift, shift)),
                             "number": (f"Animal number: {predicted['bboxes'].shape[0]}", (shift, 100)),
                             "mass": (f"Total biomass: {round(mass_total, dec)} ug", (shift, 150)),
                             "mean": (f"Animal mass mean: {round(mass_mean, dec)} ug", (shift, 200)),
                             "std": (f"Animal mass std: {round(mass_std, dec)} ug", (shift, 250))}

                for text, position in info_dict.values():
                    img = cv2.putText(img, text, position,
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                for pts in lengths_points:
                    pts_round = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
                    img = cv2.polylines(img, [pts_round], False, (0, 255, 255), 2)

                for pts in predicted["keypoints"]:
                    width_points = pts[-2:].astype(np.int32)
                    img = cv2.line(img, tuple(width_points[0]), tuple(width_points[1]),
                                   color=(0, 255, 255), thickness=2)

            else:
                print(f"{'-' * 50}"
                      f"Mass calculation results empty for file: {str(img_path)}"
                      f"{'-' * 50}")

            print(f"\nInference time: {time.time() - start}\n")

            # results_df.to_csv(args.output_dir / f"{img_path.stem}_results.csv")
            # cv2.imwrite(str(args.output_dir / f"{img_path.stem}_results.jpg"), img)

            cv2.imshow('predicted', cv2.resize(img, (1400, 700)))
            cv2.waitKey(20000)

        except Exception as e:
            print(traceback.format_exc())

    print(f"\nProcessing finished.\n")


if __name__ == '__main__':
    visualize(parse_args())
