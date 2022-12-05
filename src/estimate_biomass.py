import logging
import time

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import splprep, splev

import src.keypoints_detector.config as cfg
from src.keypoints_detector.model import keypoint_detector
from src.keypoints_detector.predict import predict
from src.scale_detector.scale_detector import read_scale
from src.keypoints_detector.utils import calc_dist


def fit_polynomial(bbox, points, keypoints_num):
    bbox_w = bbox[0] - bbox[2]
    bbox_l = bbox[1] - bbox[3]

    pts_ind = [0, 1] if bbox_w <= bbox_l else [1, 0]

    x = points[: keypoints_num - 2][:, pts_ind[0]]
    y = points[: keypoints_num - 2][:, pts_ind[1]]

    model = np.poly1d(np.polyfit(x, y, 5))
    range_pts = [x[0], x[-1]]
    lin_sp = np.arange(min(range_pts), max(range_pts), 10)
    res = [lin_sp, model(lin_sp)]
    pts = np.round(np.hstack((np.resize(res[pts_ind[0]], (res[pts_ind[0]].shape[0], 1)),
                              np.resize(res[pts_ind[1]], (res[pts_ind[1]].shape[0], 1)))))
    pts_num = pts.shape[0]

    # TODO: create different error handling function
    if pts_num == 0:
        pts = np.reshape(bbox, ((bbox.size // 2), 2))
        return pts

    list_ind = [0, pts_num - 1]
    distances = [calc_dist(pts[0], points[0]), calc_dist(pts[0], points[keypoints_num - 3])]
    position = list_ind[distances.index(min(distances))]

    insertion_ind = [keypoints_num - 3, 0] if position == 0 else [0, keypoints_num - 3]
    inserted = np.append(pts, points[insertion_ind[0]])
    inserted = np.insert(inserted, 0, points[insertion_ind[1]])

    pts = np.reshape(inserted, ((inserted.size // 2), 2))
    return pts


def fit_bspline(bbox, points, keypoints_num):
    try:
        x = points[: keypoints_num - 2][:, 0]
        y = points[: keypoints_num - 2][:, 1]
        tck, u = splprep([x, y], s=3)
        pts = splev(u, tck)
        pts = np.round(np.hstack((np.resize(pts[0], (pts[0].shape[0], 1)),
                                  np.resize(pts[1], (pts[1].shape[0], 1)))))
    except ValueError:
        pts = np.reshape(np.array(bbox), ((bbox.size // 2), 2))
        return pts

    return pts


def calc_dimensions(length_pts, width_pts, scale_ratio):
    points = width_pts.astype(np.uint64)
    right, left = points
    width = calc_dist(left, right) * scale_ratio
    len_parts = [calc_dist(length_pts[i], length_pts[i + 1]) for i in range(len(length_pts) - 1)]
    length = np.sum(np.asarray(len_parts)) * scale_ratio
    return length, width


def calculate_mass(predicted, scale, img_path, curve_fit_algorithm="bspline"):
    scale_ratio = scale["um"] / scale["pix"]
    density = 1.04
    results = []
    lengths_points = []
    fit_algorithms = {
        "bspline": fit_bspline,
        "polynomial": fit_polynomial
    }

    for i, (bbox, points) in enumerate(zip(predicted["bboxes"], predicted["keypoints"])):
        length_pts = fit_algorithms[curve_fit_algorithm](bbox, points, cfg.KEYPOINTS)
        lengths_points.append(length_pts)
        length, width = calc_dimensions(length_pts, points[-2:], scale_ratio)

        class_name = cfg.INSTANCE_CATEGORY_NAMES[predicted["labels"][i]]

        if length and width != np.nan:
            R = length / width

            # If the mass of the species Echiniscus is estimated, use different equation
            if class_name == 'heter_ech':
                mass = (1 / 12) * length * np.pi * (length / R) ** 2 * density * 10 ** -6  # [ug]
            else:
                mass = length * np.pi * (length / (2 * R)) ** 2 * density * 10 ** -6  # [ug]

            info = {"img_path": img_path,
                    "id": i,
                    "class": class_name,
                    "length": length,
                    "width": width,
                    "biomass": mass}

            # print(info)
            results.append(info)

    return pd.DataFrame(results), lengths_points


def main(args, queue):
    args.output_dir.mkdir(exist_ok=True, parents=True)
    images_extensions = ("png", "tif", "jpg", "jpeg")

    images_paths = []
    for ext in images_extensions:
        ext_paths = list(args.input_dir.glob(f"*.{ext}"))
        images_paths.extend(ext_paths)

    model = keypoint_detector()
    model.load_state_dict(torch.load("../keypoints_detector/checkpoints/keypoints_detector.pth"))
    for img_path in images_paths:
        try:
            start = time.time()
            img = cv2.imread(str(img_path), 1)
            image_scale, img = read_scale(img, device="cpu")
            predicted = predict(model, img, device=cfg.DEVICE)
            results_df, lengths_points = calculate_mass(predicted, image_scale, img_path)

            img = predicted["image"]

            if not results_df.empty:
                mass_total = results_df["biomass"].sum()
                mass_mean = results_df["biomass"].mean()
                mass_std = results_df["biomass"].std()

                logging.info(f"image path: {img_path}\n"
                             f"Image scale: {image_scale}\n"
                             f"Image total mass: {mass_total} ug\n"
                             f"{'-' * 50}")
                queue.put(F"Image processed: {str(img_path)}\n")

                info_dict = {"scale": (f"Scale: {image_scale['um']} um", (50, 50)),
                             "number": (f"Animal number: {predicted['bboxes'].shape[0]}", (50, 100)),
                             "mass": (f"Total biomass: {round(mass_total, 5)} ug", (50, 150)),
                             "mean": (f"Animal mass mean: {round(mass_mean, 5)} ug", (50, 200)),
                             "std": (f"Animal mass std: {round(mass_std, 5)} ug", (50, 250))}

                for text, position in info_dict.values():
                    img = cv2.putText(img, text, position,
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                for pts in lengths_points:
                    img = cv2.polylines(img, [pts.astype(np.int32)], False, (0, 255, 255), 2)

            else:
                logging.info(f"{'-' * 50}"
                             f"Mass calculation results empty for file: {str(img_path)}"
                             f"{'-' * 50}")
                queue.put(f"results empty for {str(img_path)}")

            logging.info(f"\nInference time: {time.time() - start}\n")

            results_df.to_csv(args.output_dir / f"{img_path.stem}_results.csv")
            cv2.imwrite(str(args.output_dir / f"{img_path.stem}_results.jpg"), img)


            # cv2.imshow('predicted', cv2.resize(img, (1400, 700)))
            # cv2.waitKey(2500)

        except Exception as e:
            print(e)
            logging.error(e)

    logging.info(f"\nProcessing finished.\n")
    queue.put("Processing finished.\n")
