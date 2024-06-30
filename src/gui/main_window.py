import sys
from collections import OrderedDict
from pathlib import Path
from queue import Queue
from threading import Thread

from PyQt5 import QtGui, uic
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (QFileDialog, QTextEdit)
from gui.correction_tool import CorrectionTool
from gui.help_window import HelpWindow
from gui.utils import MsgWorker


class UI(CorrectionTool):
    """Application functionality base class"""
    def __init__(self):
        super().__init__()
        uic.loadUi(str(self.find_file_dir("src/gui/app_window.ui")), self)
        self.initialize_scene()
        self.textbox = self.findChild(QTextEdit, "textEdit")
        self.cursor = self.textbox.textCursor()
        self.graphicsView.viewport().installEventFilter(self)
        self.help_window = HelpWindow(str(self.find_file_dir("docs/user_manual.pdf")))
        self.icon_path = Path(self.find_file_dir("src/gui/icons").parent / "icons/tarmass_icon.png")
        self.auto_save = False
        self.stop_flag = True
        self.stop_proc_threads_flag = False
        self._folder_path_in = None
        self._folder_path_out = None
        self.inference_thread = None
        self.mass_calc_thread = None
        self.report_thread = None
        self.correction_tool_thread = None
        self.msg_queue = Queue()
        self.start_msg_thread()
        self.spline_interpolation_algorithm = None
        self.inference_model_name = None

        self.category_names = OrderedDict({
            'eutardigrada black': 'eutar_bla',
            'heterotardigrada echiniscus': 'heter_ech',
            'eutardigrada translucent': 'eutar_tra',
            'scale': 'scale'
        })
        self.models_names = {
            'YOLOv8m pose': 'yolov8',
            'keypoint RCNN': 'kpt_rcnn'
        }

        self.connect_widgets()
        self.set_variables()

    @staticmethod
    def find_file_dir(file_dir):
        if getattr(sys, 'frozen', False):
            return Path(sys.executable).parent / Path(file_dir).name
        else:
            return Path.cwd().parent / file_dir

    def connect_widgets(self):
        """Connect buttons and menu actions to the functions"""
        # menu
        self.actionOpen_Dir.triggered.connect(self.select_folder_in)
        self.actionChange_Save_Dir.triggered.connect(self.select_folder_out)
        self.actionSave.setShortcut(QtGui.QKeySequence("Ctrl+s"))
        self.actionSave.triggered.connect(self.save_points)
        self.menuHelp.triggered.connect(self.open_help)

        # control tab
        self.inferenceButton.pressed.connect(self.start_inference)
        self.stopButton.pressed.connect(self.stop_processing)
        self.calculateButton.pressed.connect(self.start_calc_mass)
        self.reportButton.pressed.connect(self.start_report)
        self.clearButton.pressed.connect(self.clear_info)
        self.interpolationComboBox.currentTextChanged.connect(self.curve_algorithm_change)
        self.detectionComboBox.currentTextChanged.connect(self.detection_algorithm_change)
        # correction tool tab
        self.openImageButton.pressed.connect(self.open_image)
        self.nextButton.pressed.connect(self.next_image)
        self.previousButton.pressed.connect(self.previous_image)
        self.instanceButton.pressed.connect(self.create_instance)
        self.scaleSpinBox.valueChanged.connect(self.scale_value_change)
        self.autosaveBox.stateChanged.connect(self.auto_save_change)

    def set_variables(self):
        self.setWindowIcon(QtGui.QIcon(str(self.icon_path)))
        self.setMouseTracking(True)
        self.spline_interpolation_algorithm = self.interpolationComboBox.currentText()
        self.detection_algorithm_change(self.detectionComboBox.currentText())
        self.autosaveBox.setChecked(True)

    def textbox_print_msg(self, msg):
        """Add message to the textbox of the application's 'Control' tab"""
        self.textbox.append(msg)
        self.cursor.movePosition(QtGui.QTextCursor.End)
        self.textbox.setTextCursor(self.cursor)

    def start_msg_thread(self):
        """Run message printing worker thread"""
        self.msg_thread = MsgWorker(self.msg_queue)
        self.msg_thread.msg_signal.connect(self.textbox_print_msg)
        self.msg_thread.start()

    def select_folder_in(self):
        """Add images paths to the list and the application widget, show selected image"""
        self.images_paths = []
        self.imagesListWidget.clear()
        path = "" if self._folder_path_in is None else self._folder_path_in
        if path == "" and self._folder_path_out is not None:
            path = self._folder_path_out
        self._folder_path_in = Path(QFileDialog.getExistingDirectory(self, "Choose input directory", str(path),
                                                                     QFileDialog.ShowDirsOnly))
        images_extensions = ("png", "tif", "jpg", "jpeg")
        for ext in images_extensions:
            ext_paths = list(self._folder_path_in.glob(f"*.{ext}"))
            self.images_paths.extend(ext_paths)
            for pth in ext_paths:
                self.imagesListWidget.addItem(str(pth))

    def select_folder_out(self):
        """Set output folder for the 'Inference' and 'Calculate statistics' processes"""
        path = "" if self._folder_path_out is None else self._folder_path_out
        if path == "" and self._folder_path_in is not None:
            path = self._folder_path_in
        self._folder_path_out = Path(QFileDialog.getExistingDirectory(self, "Choose output directory", str(path),
                                                                      QFileDialog.ShowDirsOnly))

    def check_folders(self):
        """
        Check if both, input and output folders has been selected. Print message otherwise
        :return: bool
        """
        if self._folder_path_in is None:
            msg = "Input folder not defined.\n"
            self.msg_queue.put(msg)
            return False
        if self._folder_path_out is None:
            msg = "Output folder not defined.\n"
            self.msg_queue.put(msg)
            return False
        return True

    def auto_save_change(self):
        self.auto_save = not self.auto_save

    def open_help(self):
        file_url = f"file://{self.help_window.pdf_dir}"
        self.help_window.show()
        self.help_window.webView.setUrl(QUrl(file_url))

    def inference_worker(self, stop):
        """Override this method - detect tardigrades and scales in images"""
        pass

    def calc_mass_worker(self, stop):
        """Override this method - calculate tardigrades masses based on inference output"""
        pass

    def report_worker(self, stop):
        """Override this method - generate final report"""
        pass

    def start_inference(self):
        """Connected to the 'Inference' button - runs inference thread"""
        if self.stop_flag:
            if self.check_folders():
                self.inference_thread = Thread(target=self.inference_worker, daemon=True,
                                               args=(lambda: self.stop_proc_threads_flag,))
                self.inference_thread.start()
                self.stop_flag = False
                self.msg_queue.put("Inference started.\n")

    def start_calc_mass(self):
        """Connected to the 'Calculate statistics' button - runs calc mass thread"""
        if self.stop_flag:
            if self.check_folders():
                self.mass_calc_thread = Thread(target=self.calc_mass_worker, daemon=True,
                                               args=(lambda: self.stop_proc_threads_flag,))
                self.mass_calc_thread.start()
                self.stop_flag = False
                self.msg_queue.put("Calculating biomass started.\n")

    def start_report(self):
        """Connected to the 'Generate report' button - runs report thread"""
        if self.stop_flag:
            if self.check_folders():
                self.report_thread = Thread(target=self.report_worker, daemon=True,
                                            args=(lambda: self.stop_proc_threads_flag,))
                self.report_thread.start()
                self.stop_flag = False
                self.msg_queue.put("Report generation started.\n")

    def clear_info(self):
        self.textbox.clear()
        self.cursor.movePosition(QtGui.QTextCursor.End)
        self.textbox.setTextCursor(self.cursor)

    def curve_algorithm_change(self, method):
        """Connected to the 'Line fitting algorithm' combo box - changes spline interpolation algorithm"""
        self.spline_interpolation_algorithm = method

    def detection_algorithm_change(self, method):
        self.inference_model_name = self.models_names[method]

    def stop_processing(self):
        """Sends flag to stop processing for inference and mass calc threads"""
        if not self.stop_flag:
            self.stop_proc_threads_flag = True
        else:
            msg = "Processing not started.\n"
            self.msg_queue.put(msg)

    def open_image(self):
        """Connected to the 'Open selected image' button in 'Correction tool' tab -
        opens the image selected in images list widget"""
        if len(self.images_paths) > 0:
            img_path = self.imagesListWidget.currentItem()
            if img_path is not None:
                if self.auto_save and not self.first_image:
                    self.save_points()
                img_num = self.imagesListWidget.currentRow()
                self.current_image = img_num
                self.set_scene(Path(img_path.text()))
                self.first_image = False
            else:
                msg = "Image not selected.\n"
                self.msg_queue.put(msg)

    def update_img(self, i):
        self.current_image += i
        self.set_scene(self.images_paths[self.current_image])
        self.imagesListWidget.setCurrentRow(self.current_image)

    def next_image(self):
        """Connected to the 'Next' button in 'Correction tool' tab"""
        if self.auto_save:
            self.save_points()

        if self.current_image + 1 <= len(self.images_paths) - 1:
            self.update_img(1)

    def previous_image(self):
        """Connected to the 'Previous' button in 'Correction tool' tab"""
        if self.auto_save:
            self.save_points()

        if self.current_image - 1 >= 0:
            self.update_img(-1)
