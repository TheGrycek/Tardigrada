from collections import OrderedDict, defaultdict
from pathlib import Path
from queue import Queue
from random import randint
from threading import Thread

import ujson
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, QEvent, Qt
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QGraphicsItemGroup, \
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsWidget
from PyQt5.QtWidgets import QMainWindow, QDialog, QFileDialog, QTextEdit


class MsgWorker(QThread):
    msg_signal = pyqtSignal(str)

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            msg = self.queue.get()
            self.msg_signal.emit(msg)


class InstanceWindow(QDialog):
    def __init__(self):
        super(InstanceWindow, self).__init__()
        uic.loadUi("./gui/instance_window.ui", self)


class TardigradeItem(QGraphicsWidget):
    def __init__(self, parent, rectangle, keypoints):
        super(TardigradeItem, self).__init__(parent)
        self.rectangle = rectangle
        self.keypoints = keypoints


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("./gui/app_window.ui", self)
        # self.setWindowIcon(QtGui.QIcon("./gui/icon.png"))
        # print(QFile.exists("./gui/icon.png"))
        self.initialize_scene()
        self.textbox = self.findChild(QTextEdit, "textEdit")
        self.cursor = self.textbox.textCursor()
        self.graphicsView.viewport().installEventFilter(self)
        self.stop_flag = True
        self.stop_proc_threads_flag = False
        self._folder_path_in = None
        self._folder_path_out = None
        self.image_data = defaultdict(lambda: None)
        self.inference_thread = None
        self.mass_calc_thread = None
        self.correction_tool_thread = None
        self.current_image = 0
        self.images_paths = []
        self.msg_queue = Queue()
        self.start_msg_thread()

        self.point_size = 5
        self.connect_widgets()
        self.category_names = OrderedDict({'eutardigrada black': 'eutar_bla',
                                           'heterotardigrada echiniscus': 'heter_ech',
                                           'eutardigrada translucent': 'eutar_tra',
                                           'scale': 'scale'})

    def initialize_scene(self):
        self.scene = QGraphicsScene()
        self.pixmap = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

    def connect_widgets(self):
        self.actionOpen_Dir.triggered.connect(self.select_folder_in)
        self.actionChange_Save_Dir.triggered.connect(self.select_folder_out)
        self.actionSave.triggered.connect(self.save_points)
        # control tab
        self.inferenceButton.pressed.connect(self.start_inference)
        self.stopButton.pressed.connect(self.stop_processing)
        self.calculateButton.pressed.connect(self.start_calc_mass)
        # correction tool tab
        self.openImageButton.pressed.connect(self.open_image)
        self.nextButton.pressed.connect(self.next_image)
        self.previousButton.pressed.connect(self.previous_image)
        self.instanceButton.pressed.connect(self.create_instance)

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

        images_extensions = ("png", "tif", "jpg", "jpeg")
        i = 0
        for ext in images_extensions:
            ext_paths = list(self._folder_path_in.glob(f"*.{ext}"))
            self.images_paths.extend(ext_paths)
            for pth in ext_paths:
                self.imagesListWidget.addItem(str(pth))
                self.set_scene(pth, i)
                i += 1

    def select_folder_out(self):
        path = "" if self._folder_path_out is None else self._folder_path_out
        self._folder_path_out = Path(QFileDialog.getExistingDirectory(self, "Choose output directory", str(path)))

    def check_folders(self):
        if self._folder_path_in is None:
            msg = "Input folder not defined.\n"
            self.msg_queue.put(msg)
            return False
        if self._folder_path_out is None:
            msg = "Output folder not defined.\n"
            self.msg_queue.put(msg)
            return False
        return True

    def create_items_group(self, position, image_data, label_txt, item_type="rect", pen_colour=Qt.blue,
                           brush_colour=None):
        item_data = {"label": label_txt,
                     "value": image_data}
        item = QGraphicsItemGroup()
        item.setData(0, item_data)
        if item_type == "rect":
            x1, y1, x2, y2 = position
            elem = QGraphicsRectItem(x1, y1, abs(x2 - x1), abs(y2 - y1))
        else:
            elem = QGraphicsEllipseItem(position[0], position[1], self.point_size, self.point_size)

        elem.setPen(QtGui.QPen(Qt.transparent if pen_colour is None else pen_colour))
        elem.setBrush(QtGui.QBrush(Qt.transparent if brush_colour is None else brush_colour))
        label = QGraphicsTextItem()
        label.setPlainText(label_txt)
        label.setDefaultTextColor(pen_colour)
        label.setPos(position[0], position[1])
        [item.addToGroup(it) for it in (elem, label)]
        item.setFlag(QGraphicsItem.ItemIsMovable)
        item.setFlag(QGraphicsItem.ItemIsSelectable)
        item.setCursor(Qt.CrossCursor)
        return item

    def add_rect(self, bbox, class_name, data=None, colour=Qt.blue):
        rect_item = self.create_items_group(bbox, data, class_name, item_type="rect", pen_colour=colour)
        self.scene.addItem(rect_item)
        return rect_item

    def add_pt(self, pt, label_num, colour=Qt.blue):
        pt_item = self.create_items_group(pt, None, f"{label_num + 1}", item_type="point",
                                          pen_colour=colour, brush_colour=colour)
        self.scene.addItem(pt_item)
        return pt_item

    @staticmethod
    def random_qt_colour():
        colour = QtGui.QColor()
        colour.setRgb(*[randint(50, 255) for _ in range(3)])
        return colour

    def download_img_data(self, img_path):
        data_path = img_path.parent / (img_path.stem + ".json")
        if data_path.is_file():
            self.image_data = ujson.load(data_path.open("rb"))
            colour = self.random_qt_colour()
            self.add_rect(self.image_data["scale_bbox"], "scale", self.image_data["scale_value"], colour=colour)
            category_list = list(self.category_names.values())
            for i, annot in enumerate(self.image_data["annotations"]):
                colour = self.random_qt_colour()
                class_name = category_list[annot["label"] - 1]
                bbox, kpts = annot["bbox"], annot["keypoints"]
                rect_item = self.add_rect(bbox, class_name, colour=colour)
                points = [self.add_pt(pt, i, colour=colour) for i, pt in enumerate(kpts)]
                tard_item = TardigradeItem(None, rect_item, points)
                tard_item.setData(0, {"label": class_name,
                                      "value": i})
                self.scene.addItem(tard_item)

    @staticmethod
    def get_position(obj):
        x1, y1, x2, y2 = obj.shape().controlPointRect().getCoords()
        dx, dy = obj.scenePos().x(), obj.scenePos().y()
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def set_scene(self, img_path: Path, img_num=None):
        self.initialize_scene()
        self.download_img_data(img_path)
        img = QtGui.QPixmap(str(img_path))
        self.pixmap.setPixmap(img)
        self.graphicsView.setScene(self.scene)
        if img_num is not None:
            self.current_image = img_num

    def inference_worker(self, stop):
        pass

    def calc_mass_worker(self, stop):
        pass

    def correction_tool_worker(self, stop):
        pass

    def start_inference(self):
        if self.stop_flag:
            if self.check_folders():
                self.inference_thread = Thread(target=self.inference_worker, daemon=True,
                                               args=(lambda: self.stop_proc_threads_flag,))
                self.inference_thread.start()
                self.stop_flag = False
                self.msg_queue.put("Processing started.\n")

    def start_calc_mass(self):
        if self.stop_flag:
            if self.check_folders():
                self.mass_calc_thread = Thread(target=self.calc_mass_worker, daemon=True,
                                               args=(lambda: self.stop_proc_threads_flag,))
                self.mass_calc_thread.start()
                self.stop_flag = False
                self.msg_queue.put("Processing started.\n")

    def stop_processing(self):
        if not self.stop_flag:
            self.stop_proc_threads_flag = True
        else:
            msg = "Processing not started.\n"
            self.msg_queue.put(msg)

    def open_image(self):
        if len(self.images_paths) > 0:
            img_path = self.imagesListWidget.currentItem().text()
            img_num = self.imagesListWidget.currentRow()
            self.set_scene(Path(img_path), img_num=img_num)

    def update_img(self):
        self.set_scene(self.images_paths[self.current_image], img_num=self.current_image)
        self.imagesListWidget.setCurrentRow(self.current_image)

    def next_image(self):
        if self.current_image + 1 <= len(self.images_paths) - 1:
            self.current_image += 1
            self.update_img()

    def previous_image(self):
        if self.current_image - 1 >= 0:
            self.current_image -= 1
            self.update_img()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel and source == self.graphicsView.viewport() and \
                event.modifiers() == Qt.ControlModifier:
            scale = 1.20 if event.angleDelta().y() > 0 else 0.8
            self.graphicsView.scale(scale, scale)
            return True

        if event.type() == QEvent.MouseButtonPress and source == self.graphicsView.viewport() and \
                event.modifiers() == Qt.ControlModifier:
            point = self.graphicsView.mapToScene(event.pos())
            self.daw_element([point.x(), point.y()], elem_type="point")

        return super().eventFilter(source, event)

    def daw_element(self, data: list, elem_type="point"):
        # TODO: Modify this function
        if elem_type == "point":
            x, y = data
            pos_x = round(x - self.point_size / 2)
            pos_y = round(y - self.point_size / 2)
            pt = self.scene.addEllipse(pos_x, pos_y, self.point_size, self.point_size, self.pen_red, self.brush_red)
            pt.setFlag(QGraphicsItem.ItemIsMovable)
            pt.setFlag(QGraphicsItem.ItemIsSelectable)
            return pt

        elif elem_type == "rectangle":
            x1, y1, x2, y2 = data
            w = round(abs(x1 - x2))
            h = round(abs(y1 - y2))
            rect = self.scene.addRect(round(x1), round(y1), w, h, self.pen_blue, self.brush_transparent)
            rect.setFlag(QGraphicsItem.ItemIsMovable)
            rect.setFlag(QGraphicsItem.ItemIsSelectable)
            return rect

    def create_instance(self):
        self.window = InstanceWindow()
        self.window.show()
        if self.window.exec():
            instance_class = self.window.comboBox.currentText()
            self.create_object_instance(self.category_names[instance_class])

    def create_object_instance(self, class_name):
        # TODO finish this function
        pass

    def save_points(self):
        img_path = self.images_paths[self.current_image]
        data_path = img_path.parent / (img_path.stem + ".json")
        img_data = {"path": str(img_path)}
        category_list = list(self.category_names.values())
        annotations = []

        for item in self.scene.items():
            if isinstance(item, TardigradeItem):
                bbox = self.get_position(item.rectangle)
                kpts = [self.get_position(pt)[:2] for pt in item.keypoints]
                annotations.append({
                    "label": category_list.index(item.data(0)["label"]) + 1,
                    "bbox": bbox,
                    "keypoints": kpts})
            elif item.data(0) is not None and item.data(0)["label"] == "scale":
                img_data["scale_value"] = item.data(0)["value"]
                img_data["scale_bbox"] = self.get_position(item)

        img_data["annotations"] = annotations
        ujson.dump(img_data, data_path.open("w"))
        self.msg_queue.put(f"Changes saved to the file {str(data_path)}.\n")
