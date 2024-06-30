import ast
import shutil

import cv2
import keypoints_detector.config as cfg
import numpy as np
import pandas as pd
import ujson
from estimate_biomass import prepare_paths
from report.pdf import ReportPDF


def draw_bbox_with_label(img, bbox, label, colour=(255, 0, 0)):
    pt1 = tuple(bbox[:2].astype(np.uint32))
    pt2 = tuple(bbox[2:].astype(np.uint32))
    position = (int(bbox[0]), int(bbox[1]))

    img = cv2.rectangle(img.copy(), pt1, pt2, colour, 2)
    img = cv2.putText(img, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.7, color=colour, thickness=2)

    return img


def generate_images(result_df, report_path, queue, stop, args):
    images_paths = prepare_paths(args)
    for img_path in images_paths:
        if stop():
            queue.put("Processing stopped.\n")
            return

        img = cv2.imread(str(img_path))
        json_path = args.output_dir / f"{img_path.stem}.json"
        with json_path.open("r") as json_f:
            json_file = ujson.load(json_f)

        df = result_df[result_df["img_path"] == str(img_path)]

        biomass = df["biomass"].sum()
        mean = df["biomass"].mean()
        std = df["biomass"].std()

        shift, dec = 50, 5  # shift the text from the border, round to 5 decimal places
        info_dict = {"scale": (f"Scale: {json_file['scale_value']} um", (shift, shift)),
                     "number": (f"Animal number: {len(df)}", (shift, 100)),
                     "mass": (f"Total biomass: {round(biomass, dec)} ug", (shift, 150)),
                     "mean": (f"Animal mass mean: {round(mean, dec)} ug", (shift, 200)),
                     "std": (f"Animal mass std: {round(std, dec)} ug", (shift, 250))}

        scale_bbox = np.array(json_file["scale_bbox"])
        img = draw_bbox_with_label(img, scale_bbox, "scale", colour=(0, 255, 0))
        for text, position in info_dict.values():
            img = cv2.putText(img, text, position,
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        for points in df["fitted_points"].to_numpy():
            pts_round = np.round(points).astype(np.int32).reshape((-1, 1, 2))
            img = cv2.polylines(img, [pts_round], False, (0, 255, 255), 2)

        for annot in json_file["annotations"]:
            width_points = np.array(annot["keypoints"][-2:]).astype(np.int32)
            img = cv2.line(img, tuple(width_points[0]), tuple(width_points[1]),
                           color=(0, 255, 255), thickness=2)
            bbox = np.array(annot["bbox"])
            label = cfg.INSTANCE_CATEGORY_NAMES[int(annot["label"])]
            img = draw_bbox_with_label(img, bbox, label, colour=(255, 0, 0))

        cv2.imwrite(str(report_path / "images" / f"{img_path.stem}_results.jpg"), img)
        queue.put(f"Image processed: {str(img_path)}\n")


def generate_report(args, icon_path, queue, stop):
    report_path = args.output_dir / "report"
    if report_path.is_dir():
        shutil.rmtree(str(report_path))
    report_path.mkdir(parents=True)
    (report_path / "images").mkdir(parents=True, exist_ok=True)

    result_df = pd.read_csv(args.output_dir / "results.csv")
    result_df.fitted_points = result_df.fitted_points.apply(lambda x: np.array(ast.literal_eval(x)))

    generate_pdf(result_df, report_path, icon_path, queue)
    generate_images(result_df, report_path, queue, stop, args)


def generate_pdf(df, report_path, icon_path, queue):
    pdf = ReportPDF(df, report_path, icon_path)
    pdf.generate()
    pdf.output(str(report_path / 'BiomassReport.pdf'))
    queue.put("Report generated.\n")
