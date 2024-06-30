from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QGraphicsWidget


class MsgWorker(QThread):
    """Worker for sending messages to the message queue (presented in UI.textbox)"""
    msg_signal = pyqtSignal(str)

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            msg = self.queue.get()
            self.msg_signal.emit(msg)


class InstanceWindow(QDialog):
    """Window for screen object selection (scale, tardigrade black,...)"""
    def __init__(self, file_dir):
        super(InstanceWindow, self).__init__()
        uic.loadUi(file_dir, self)


class TardigradeItem(QGraphicsWidget):
    """Widget representing single tardigrade"""
    def __init__(self, label, rectangle, keypoints):
        super().__init__()
        self.label = label
        self.rectangle = rectangle
        self.keypoints = keypoints
        self.set_data()

    def set_data(self):
        for kpt in self.keypoints:
            kpt.setData(1, self)

        self.label.setData(1, self)
        self.rectangle.setData(1, self)

    def get_label(self):
        return self.label

    def get_rectangle(self):
        return self.rectangle

    def get_keypoints(self):
        return self.keypoints

    def set_z_value(self, z_value):
        for kpt in self.keypoints:
            kpt.setZValue(z_value)

        self.label.setZValue(z_value)
        self.rectangle.setZValue(z_value)
