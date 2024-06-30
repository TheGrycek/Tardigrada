from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fpdf import FPDF


class ReportPDF(FPDF):
    def __init__(self, df, report_path, icon_path):
        super().__init__()
        self.width = 210
        self.height = 297
        self.df = df
        self.font1 = 'Helvetica'
        self.font2 = 'Times'
        self.report_path = Path(report_path)
        self.plots_path = self.report_path / "plots"
        self.plots_path.mkdir(exist_ok=True, parents=True)
        self.icon_path = icon_path

    def header(self):
        self.image(str(self.icon_path), 10, 8, 16)
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
        text = (f"Folder: {stats['folder']} \n"
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

    @staticmethod
    def get_bins_num(data):
        quartiles = data.quantile([0.25, 0.75])
        iqr = quartiles[0.75] - quartiles[0.25]
        bin_width = 2 * (iqr / (len(data) ** (1 / 3)))
        bins_num = (data.max() - data.min()) / bin_width
        return int(bins_num)

    def create_plot(self, kind, class_name, unit=None):
        plot_args = {
            "title": f"{class_name} count",
            "figsize": (20, 16),
            "fontsize": 32,
            "grid": True
        }

        if kind == "bar":
            data = self.df[class_name].value_counts()
            fig = data.plot(kind=kind, rot=0, **plot_args)

        elif kind == "hist":
            data = self.df[class_name]
            fig = data.plot.hist(bins=self.get_bins_num(data), alpha=1.0, **plot_args)
            plt.xlabel(f"{class_name} [{unit}]", fontsize=26)

        plt.ylabel("occurrence", fontsize=26)
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

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
            "folder": str(Path(self.df.img_path.iloc[0]).name),
            "img_num": len(self.df.img_path.unique()),
            "animal_sum": len(self.df),
            "biomass_sum": round(self.df.biomass.sum(), 3),
            "biomass_mean": round(self.df.biomass.mean(), 3),
            "biomass_std": round(self.df.biomass.std(), 3),
            "length_mean": round(self.df.length.mean(), 3),
            "length_std": round(self.df.length.std(), 3),
            "width_mean": round(self.df.width.mean(), 3),
            "width_std": round(self.df.width.std(), 3),
            "classes_plot_dir": classes_plot_dir,
            "biomass_plot_dir": biomass_plot_dir,
            "length_plot_dir": length_plot_dir,
            "width_plot_dir": width_plot_dir
        }

        return output
