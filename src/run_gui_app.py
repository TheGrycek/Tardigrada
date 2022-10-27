#!/usr/bin/env python3
import logging
import sys
from collections import namedtuple

from PyQt5.QtWidgets import QApplication

from estimate_biomass import run_inference, run_calc_mass
from gui.main_window import Window

logging.basicConfig(level=logging.DEBUG, filename="logging.log")


class AppWindow(Window):
    def __init__(self):
        super().__init__()

    def inference_worker(self, stop):
        args = namedtuple("args", ["input_dir", "output_dir"])
        args_parsed = args(self._folder_path_in, self._folder_path_out)
        run_inference(args_parsed, self.msg_queue, stop)
        self.stop_flag = True
        self.stop_proc_threads_flag = False

    def calc_mass_worker(self, stop):
        args = namedtuple("args", ["output_dir"])
        args_parsed = args(self._folder_path_out)
        run_calc_mass(args_parsed, self.msg_queue, stop)
        self.stop_flag = True
        self.stop_proc_threads_flag = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    app.exec_()
