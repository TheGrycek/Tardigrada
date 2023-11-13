#!/usr/bin/env python3
import logging
import sys
from collections import namedtuple

from PyQt5.QtWidgets import QApplication

from estimate_biomass import run_inference, run_calc_mass
from report.report import generate_report
from gui.main_window import UI
from report.report import generate_report

logging.basicConfig(level=logging.DEBUG, filename="logging.log")


class MainWindow(UI):
    def __init__(self):
        super().__init__()

    def inference_worker(self, stop):
        """Runs in self.inference_thread"""
        args = namedtuple("args", ["input_dir", "output_dir"])
        args_parsed = args(self._folder_path_in, self._folder_path_out)
        run_inference(args_parsed, self.msg_queue, stop, self.inference_model_name)
        self.stop_flag = True
        self.stop_proc_threads_flag = False

    def calc_mass_worker(self, stop):
        """Runs in self.mass_calc_thread"""
        args = namedtuple("args", ["output_dir"])
        args_parsed = args(self._folder_path_out)
        run_calc_mass(args_parsed, self.msg_queue, stop, self.spline_interpolation_algorithm)
        self.stop_flag = True
        self.stop_proc_threads_flag = False

    def report_worker(self, stop):
        """Runs in self.report_thread"""
        args = namedtuple("args", ["input_dir", "output_dir"])
        args_parsed = args(self._folder_path_in, self._folder_path_out)
        generate_report(args_parsed, self.icon_path, self.msg_queue, stop)
        self.stop_flag = True
        self.stop_proc_threads_flag = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
