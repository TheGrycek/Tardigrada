from pathlib import Path
from queue import Queue
from threading import Thread

import matplotlib.pyplot as plt
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QPushButton, QFileDialog, QTextEdit, QMainWindow


class MsgWorker(QThread):
    msg_signal = pyqtSignal(str)

    def __init__(self, queue):
        super().__init__()
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
        self.button_inference_start = QPushButton("Inference", self)
        self.button_calc_mass_start = QPushButton("Calc mass", self)
        self.button_stop = QPushButton("Stop", self)
        self._folder_path_in = None
        self._folder_path_out = None
        self.stop_flag = True
        self.stop_proc_threads_flag = False
        self.msg_queue = Queue()
        self.init_window()
        self.start_msg_thread()
        self.infererence_thread = None
        self.mass_calc_thread = None

    def init_window(self):
        self.setWindowTitle('TarMass')
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.button_select_in.pressed.connect(self.select_folder_in)
        self.button_select_out.pressed.connect(self.select_folder_out)
        self.button_inference_start.pressed.connect(self.start_inference)
        self.button_stop.pressed.connect(self.stop_processing)
        self.button_calc_mass_start.pressed.connect(self.start_calc_mass)
        self.textbox.setGeometry(0, 200, 500, 100)
        self.button_select_in.setGeometry(0, 0, 150, 20)
        self.button_select_out.setGeometry(0, 20, 150, 20)
        self.button_inference_start.setGeometry(150, 0, 150, 20)
        self.button_calc_mass_start.setGeometry(150, 20, 150, 20)
        self.button_stop.setGeometry(300, 0, 150, 20)

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

    def select_folder_out(self):
        path = "" if self._folder_path_out is None else self._folder_path_out
        self._folder_path_out = Path(QFileDialog.getExistingDirectory(self, "Choose output directory", str(path)))

    def inference_worker(self, stop):
        pass

    def calc_mass_worker(self, stop):
        pass

    def start_inference(self):
        if self.stop_flag:
            if self._folder_path_in is None:
                msg = "Input folder not defined.\n"
                self.msg_queue.put(msg)
            if self._folder_path_out is None:
                msg = "Output folder not defined.\n"
                self.msg_queue.put(msg)
            else:
                self.infererence_thread = Thread(target=self.inference_worker, daemon=True,
                                                 args=(lambda: self.stop_proc_threads_flag, ))
                self.infererence_thread.start()
                self.stop_flag = False
                self.msg_queue.put("Processing started.\n")

    def start_calc_mass(self):
        if self.stop_flag:
            if self._folder_path_out is None:
                msg = "Output folder not defined.\n"
                self.msg_queue.put(msg)
            else:
                self.mass_calc_thread = Thread(target=self.calc_mass_worker, daemon=True,
                                               args=(lambda: self.stop_proc_threads_flag, ))
                self.mass_calc_thread.start()
                self.stop_flag = False
                self.msg_queue.put("Processing started.\n")

    def stop_processing(self):
        if not self.stop_flag:
            self.stop_proc_threads_flag = True
        else:
            msg = "Processing not started.\n"
            self.msg_queue.put(msg)
