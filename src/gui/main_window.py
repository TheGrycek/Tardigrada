import logging
import sys
from collections import namedtuple
from pathlib import Path
from queue import Queue
from threading import Thread

import matplotlib.pyplot as plt
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QFileDialog, QTextEdit, QMainWindow

from src.estimate_biomass import main


class MsgWorker(QThread):
    msg_signal = pyqtSignal(str)

    def __init__(self, queue):
        super(MsgWorker, self).__init__()
        self.queue = queue

    def run(self):
        while True:
            msg = self.queue.get()
            self.msg_signal.emit(msg)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(400, 400, 500, 300)
        self.textbox = QTextEdit(self)
        self.textbox.setReadOnly(True)
        self.cursor = self.textbox.textCursor()
        self.figure = plt.figure()
        self.button_select_in = QPushButton("Select input folder", self)
        self.button_select_out = QPushButton("Select output folder", self)
        self.button_start = QPushButton("Process", self)
        self.button_stop = QPushButton("Stop processing", self)
        self._folder_path_in = None
        self._folder_path_out = None
        self._stop_flag = True
        self.msg_queue = Queue()
        logging.basicConfig(level=logging.DEBUG, filename="logging.log")
        self.init_window()
        self.msgThread = QThread()
        self.start_msg_thread()

    def init_window(self):
        self.setWindowTitle('TarMass')
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.button_select_in.pressed.connect(self.select_folder_in)
        self.button_select_out.pressed.connect(self.select_folder_out)
        self.button_start.pressed.connect(self.start_processing)
        self.button_stop.pressed.connect(self.stop_processing)
        self.textbox.setGeometry(0, 200, 500, 100)
        self.button_select_in.setGeometry(0, 0, 150, 20)
        self.button_select_out.setGeometry(0, 20, 150, 20)
        self.button_start.setGeometry(150, 0, 150, 20)
        self.button_stop.setGeometry(150, 20, 150, 20)

    def textbox_print_msg(self, msg):
        self.textbox.append(msg)
        self.cursor.movePosition(QtGui.QTextCursor.End)
        self.textbox.setTextCursor(self.cursor)

    def start_msg_thread(self):
        self.msg_thread = MsgWorker(self.msg_queue)
        self.msg_thread.msg_signal.connect(self.textbox_print_msg)
        self.msg_thread.start()

    def select_folder_in(self):
        path = "" if self._folder_path_in is None else self._folder_path_in
        self._folder_path_in = Path(QFileDialog.getExistingDirectory(self, "Choose input directory", str(path)))
        logging.debug(self._folder_path_in)

    def select_folder_out(self):
        path = "" if self._folder_path_out is None else self._folder_path_out
        self._folder_path_out = Path(QFileDialog.getExistingDirectory(self, "Choose output directory", str(path)))
        logging.debug(self._folder_path_out)

    def main_process(self):
        args = namedtuple("args", ["input_dir", "output_dir"])
        args_parsed = args(self._folder_path_in, self._folder_path_out)
        main(args_parsed, self.msg_queue)
        self._stop_flag = True

    def start_processing(self):
        if self._stop_flag:
            if self._folder_path_in is None:
                msg = "Input folder not defined.\n"
                logging.warning(msg)
                self.msg_queue.put(msg)
            if self._folder_path_out is None:
                msg = "Output folder not defined.\n"
                logging.warning(msg)
                self.msg_queue.put(msg)
            else:
                self.processing_thread = Thread(target=self.main_process, daemon=True)
                self.processing_thread.start()
                self._stop_flag = False
                self.msg_queue.put("Processing started.\n")
        else:
            msg = "Processing already started.\n"
            logging.warning(msg)
            self.msg_queue.put(msg)

    def stop_processing(self):
        if not self._stop_flag:
            pass
        else:
            msg = "Processing not started.\n"
            logging.warning(msg)
            self.msg_queue.put(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec_()

