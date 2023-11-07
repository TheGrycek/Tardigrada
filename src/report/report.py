import ast
import shutil
from collections import namedtuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ujson
from fpdf import FPDF

import keypoints_detector.config as cfg
from estimate_biomass import prepare_paths


class ReportPDF(FPDF):
    def __init__(self, df, report_path):
        super().__init__()
        self.width = 210
        self.height = 297
        self.df = df
        self.font1 = 'Helvetica'
        self.font2 = 'Times'
        self.report_path = Path(report_path)
        self.plots_path = self.report_path / "plots"
        self.plots_path.mkdir(exist_ok=True, parents=True)

    def header(self):
        icon_path = cfg.REPO_ROOT / "src/gui/icons/tarmass_icon.png"
        self.image(str(icon_path), 10, 8, 33)
        self.set_font(self.font1, 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Biomass report', 0, align='C')
        self.ln(20)

    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font(self.font1, 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', border=0, align='C')

    def page_body(self):
        self.set_font(self.font2, '', 8)
        stats = self.calculate_stats()

        cell_args = {"align": "L", "w": 0, "h": 4, "border": 0}
        text = (f"Folder path: {stats['folder']} \n"
                f"Images number: {stats['img_num']} \n"
                f"Animal number: {stats['animal_sum']} \n"
                f"Total biomass: {stats['biomass_sum']} [µg] \n"
                f"Animal biomass mean: {stats['biomass_mean']} [µg] \n"
                f"Animal biomass std: {stats['biomass_std']} [µg] \n"
                f"Animal length mean: {stats['length_mean']} [µm] \n"
                f"Animal length std: {stats['length_std']} [µm] \n"
                f"Animal width mean: {stats['width_mean']} [µm] \n"
                f"Animal width std: {stats['width_std']} [µm] \n")
        self.multi_cell(text=text, **cell_args)

        image_args = {"w": 120, "h": 80, "x": int(self.width / 2) - 60}
        self.image(stats["classes_plot_dir"], **image_args)
        self.image(stats["biomass_plot_dir"], **image_args)
        self.image(stats["length_plot_dir"], **image_args)
        self.image(stats["width_plot_dir"], **image_args)

    def generate(self):
        self.alias_nb_pages()
        self.add_page()
        self.page_body()

    def create_plot(self, kind, class_name, unit=None):
        plot_args = {
            "title": f"{class_name} count",
            "figsize": (20, 16),
            "fontsize": 26,
            "grid": True
        }

        if kind == "bar":
            data = self.df[class_name].value_counts()
            fig = data.plot(kind=kind, rot=0, **plot_args)

        elif kind == "hist":
            data = self.df[class_name]
            fig = data.plot.hist(bins=len(data), alpha=1.0, **plot_args)
            plt.xlabel(f"{class_name} [{unit}]", fontsize=26)

        plt.ylabel("occurrence", fontsize=26)

        fig.title.set_size(32)
        plot_dir = str(self.plots_path / f'{class_name}_plot.png')
        fig.get_figure().savefig(plot_dir)
        fig.cla()

        return plot_dir

    def calculate_stats(self):
        classes_plot_dir = self.create_plot("bar", "class")
        biomass_plot_dir = self.create_plot("hist", "biomass", "µg")
        length_plot_dir = self.create_plot("hist", "length", "µm")
        width_plot_dir = self.create_plot("hist", "width", "µm")

        output = {
            "folder": str(Path(self.df.img_path.iloc[0]).parent),
            "img_num": len(self.df.img_path.unique()),
            "animal_sum": len(self.df),
            "biomass_sum": self.df.biomass.sum(),
            "biomass_mean": self.df.biomass.mean(),
            "biomass_std": self.df.biomass.std(),
            "length_mean": self.df.length.mean(),
            "length_std": self.df.length.std(),
            "width_mean": self.df.width.mean(),
            "width_std": self.df.width.std(),
            "classes_plot_dir": classes_plot_dir,
            "biomass_plot_dir": biomass_plot_dir,
            "length_plot_dir": length_plot_dir,
            "width_plot_dir": width_plot_dir
        }

        return output


def draw_bbox_with_label(img, bbox, label, colour=(255, 0, 0)):
    pt1 = tuple(bbox[:2].astype(np.uint32))
    pt2 = tuple(bbox[2:].astype(np.uint32))
    position = (int(bbox[0]), int(bbox[1]))

    img = cv2.rectangle(img.copy(), pt1, pt2, colour, 2)
    img = cv2.putText(img, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.7, color=colour, thickness=2)

    return img


def generate_pdf(df, report_path, queue):
    pdf = ReportPDF(df, report_path)
    pdf.generate()
    pdf.output(str(report_path / 'BiomassReport.pdf'))
    queue.put("Report generated.\n")


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


def generate_report(args, queue, stop):
    report_path = args.output_dir / "report"
    if report_path.is_dir():
        shutil.rmtree(str(report_path))
    report_path.mkdir(parents=True)
    (report_path / "images").mkdir(parents=True, exist_ok=True)

    result_df = pd.read_csv(args.output_dir / "results.csv")
    result_df.fitted_points = result_df.fitted_points.apply(lambda x: np.array(ast.literal_eval(x)))

    generate_pdf(result_df, report_path, queue)
    generate_images(result_df, report_path, queue, stop, args)


if __name__ == "__main__":
    args = namedtuple("args", ["input_dir", "output_dir"])
    args_parsed = args(
        Path("/keypoints_detector/datasets/test_ui"),
        Path("/keypoints_detector/datasets/test_ui")
    )
    generate_report(args_parsed, None, None)

