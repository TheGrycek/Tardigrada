import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from queue import Queue
from random import randint
from threading import Thread

import numpy as np
import ujson
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QFileDialog, QTextEdit, QMainWindow
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QGraphicsItemGroup, \
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsWidget

# from src.keypoints_detector.config import REPO_ROOT


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
    def __init__(self):
        super(InstanceWindow, self).__init__()
        uic.loadUi("./gui/instance_window.ui", self)


class TardigradeItem(QGraphicsWidget):
    """Widget representing single tardigrade"""
    def __init__(self, label, rectangle, keypoints):
        super().__init__()
        self.label = label
        self.rectangle = rectangle
        self.keypoints = keypoints

        for kpt in keypoints:
            kpt.setData(1, self)

        label.setData(1, self)
        rectangle.setData(1, self)

    def get_label(self):
        return self.label

    def get_rectangle(self):
        return self.rectangle

    def get_keypoints(self):
        return self.keypoints


# TODO: consider dividing into two smaller classes
class UI(QMainWindow):
    """Application functionality base class"""
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("./gui/app_window.ui", self)
        self.setWindowIcon(QtGui.QIcon("./gui/icons/bug--pencil.png"))
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
        self.follow_mouse_flag = False
        self.create_instance_flag = False
        self.setMouseTracking(True)
        self.tard_num = 0
        self.memory_item = None
        self.memory_rect_pts = None

        self.point_size = 5
        self.connect_widgets()
        self.category_names = OrderedDict({
            'eutardigrada black': 'eutar_bla',
            'heterotardigrada echiniscus': 'heter_ech',
            'eutardigrada translucent': 'eutar_tra',
            'scale': 'scale'
        })

    def initialize_scene(self):
        self.scene = QGraphicsScene()
        self.pixmap = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

    def connect_widgets(self):
        """Connect buttons and menu actions to the functions"""
        # menu
        self.actionOpen_Dir.triggered.connect(self.select_folder_in)
        self.actionChange_Save_Dir.triggered.connect(self.select_folder_out)
        self.actionSave.setShortcut(QtGui.QKeySequence("Ctrl+s"))
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
        """Add message to the textbox of the application's 'Control' tab"""
        self.textbox.append(msg)
        self.cursor.movePosition(QtGui.QTextCursor.End)
        self.textbox.setTextCursor(self.cursor)

    def start_msg_thread(self):
        """Run message printing worker thread"""
        self.msg_thread = MsgWorker(self.msg_queue)
        self.msg_thread.msg_signal.connect(self.textbox_print_msg)
        self.msg_thread.start()

    def set_scene(self, img_path):
        """
        Add an image to the 'Correction Tool' scene and draw object associated with the passed image path
        :param img_path: Path - path from which the images are downloaded
        """
        self.initialize_scene()
        self.create_img_objects(img_path)
        self.pixmap.setPixmap(QtGui.QPixmap(str(img_path)))
        self.graphicsView.setScene(self.scene)

    def select_folder_in(self):
        """Add images paths to the list and the application widget, show selected image"""
        self.images_paths = []
        self.imagesListWidget.clear()
        path = "" if self._folder_path_in is None else self._folder_path_in
        self._folder_path_in = Path(QFileDialog.getExistingDirectory(self, "Choose input directory", str(path)))
        images_extensions = ("png", "tif", "jpg", "jpeg")
        i = 0
        for ext in images_extensions:
            ext_paths = list(self._folder_path_in.glob(f"*.{ext}"))
            self.images_paths.extend(ext_paths)
            for pth in ext_paths:
                self.imagesListWidget.addItem(str(pth))
                self.current_image = i
                self.set_scene(pth)
                i += 1

    def select_folder_out(self):
        """Set output folder for the 'Inference' and 'Calculate statistics' processes"""
        path = "" if self._folder_path_out is None else self._folder_path_out
        self._folder_path_out = Path(QFileDialog.getExistingDirectory(self, "Choose output directory", str(path)))

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

    def inference_worker(self, stop):
        """Override this method - detect tardigrades and scales in images"""
        pass

    def calc_mass_worker(self, stop):
        """Override this method - calculate tardigrades masses based on inference output"""
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

    def stop_processing(self):
        """Sends flag to stop processing for inference and mass calc threads"""
        if not self.stop_flag:
            self.stop_proc_threads_flag = True
        else:
            msg = "Processing not started.\n"
            self.msg_queue.put(msg)

    def open_image(self):
        """Connected to the 'Open selected image' button in 'Correction tool' tab -
        opends image selected in images list widget"""
        if len(self.images_paths) > 0:
            img_path = self.imagesListWidget.currentItem().text()
            img_num = self.imagesListWidget.currentRow()
            self.current_image = img_num
            self.set_scene(Path(img_path))

    def update_img(self, i):
        self.current_image += i
        self.set_scene(self.images_paths[self.current_image])
        self.imagesListWidget.setCurrentRow(self.current_image)

    def next_image(self):
        """Connected to the 'Next' button in 'Correction tool' tab"""
        if self.current_image + 1 <= len(self.images_paths) - 1:
            self.update_img(1)

    def previous_image(self):
        """Connected to the 'Previous' button in 'Correction tool' tab"""
        if self.current_image - 1 >= 0:
            self.update_img(-1)

    def check_for_item(self, item_class):
        """Finds an object of the specified type in the list of selected scene items"""
        items = self.scene.selectedItems()
        if len(items) > 0:
            item = items[0]
            if isinstance(item, item_class):
                return item
        return False

    def keyPressEvent(self, event):
        """Handles key press events"""
        if event.key() == Qt.Key_Delete:
            # delete selected tadigrade item and all children items
            sel_items = self.scene.selectedItems()
            for item in sel_items:
                tard_item = item.data(1)
                if tard_item:
                    for pt in tard_item.get_keypoints():
                        self.scene.removeItem(pt)
                    self.scene.removeItem(tard_item.get_rectangle())
                    self.scene.removeItem(tard_item.get_label())
                    self.scene.removeItem(tard_item)

    def get_mouse_pos(self, event):
        """
        :param event: QEvent
        :return: tuple - tuple of floating point (x, y) mouse position
        """
        mouse_pos = self.graphicsView.mapToScene(event.pos())
        return mouse_pos.x(), mouse_pos.y()

    def resize_rect(self, event, item):
        """
        Update selected rectangle shape based on the mouse move event.
        :param event: QEvent.MouseMove - event with a mouse position
        :param item: QGraphicsRectItem - rectangle item to be resized
        :return: tuple - (x, y, w, h) rectangle coordinates
        """
        mouse_x, mouse_y = self.get_mouse_pos(event)
        dx, dy = item.scenePos().x(), item.scenePos().y()
        x1m, y1m, wm, hm = self.memory_rect_pts
        x, y = x1m, y1m

        # resize rectangle in every direction
        if mouse_x > x1m and mouse_y < y1m:
            x, y = x1m, mouse_y
        elif mouse_x < x1m and mouse_y < y1m:
            x, y = mouse_x, mouse_y
        elif mouse_x < x1m and mouse_y > y1m:
            x, y = mouse_x, y1m

        w = abs((x + dx) - mouse_x)
        h = abs((y + dy) - mouse_y)

        item.setRect(x, y, w, h)
        return [x, y, w, h]

    def update_keypoints(self, keypoints, rect):
        """
        Update keypoints position in newly created tardigrade instance rectangle
        :param keypoints: list - list of QGraphicsEllipseItem
        :param rect: tuple - (x, y, w, h) rectangle coordinates
        """
        x, y, w, h = rect
        dx = x - self.memory_rect_pts[0]
        dy = y - self.memory_rect_pts[1]
        # TODO: refactor code
        if w > h:
            l_xs_range = np.array([w * 0.05, w * 0.95])
            w_ys_range = np.array([(h * 0.2), (h * 0.8)])

            if dx < 0 and dy == 0:
                l_xs_range *= - 1
                w_ys_range = w_ys_range[::-1]
            elif dx < 0 and dy < 0:
                l_xs_range *= -1
                w_ys_range *= -1
            else:
                w_ys_range += dy

            l_xs = np.linspace(*l_xs_range, 5).reshape(-1, 1)
            w_ys = np.linspace(*w_ys_range, 2).reshape(-1, 1)
            l_ys = np.ones_like(l_xs) * (h / 2) + dy
            w_xs = np.ones_like(w_ys) * l_xs[3, 0]

        else:
            l_ys_range = np.array([h * 0.05, h * 0.95])
            w_xs_range = np.array([(w * 0.8), (w * 0.2)])

            if dx == 0 and dy < 0:
                l_ys_range *= -1
                w_xs_range = w_xs_range[::-1]

            elif dx < 0 and dy < 0:
                l_ys_range *= -1
                w_xs_range *= -1
            else:
                w_xs_range += dx

            l_ys = np.linspace(*l_ys_range, 5).reshape(-1, 1)
            w_xs = np.linspace(*w_xs_range, 2).reshape(-1, 1)
            l_xs = np.ones_like(l_ys) * (w / 2) + dx
            w_ys = np.ones_like(w_xs) * l_ys[3, 0]

        l_pts = np.concatenate((l_xs, l_ys), axis=1)
        w_pts = np.concatenate((w_xs, w_ys), axis=1)

        for kpt, new_pos in zip(keypoints, np.concatenate((l_pts, w_pts))):
            kpt.setPos(*new_pos)

        self.update()

    def eventFilter(self, source, event):
        """
        Handles mouse press, move and scroll events
        :param source: PyQt5.QtWidgets.QWidget
        :param event: QEvent
        :return: bool
        """
        # update rectangle position after double click
        if event.type() == QEvent.MouseMove and self.follow_mouse_flag:
            item = self.check_for_item(QGraphicsRectItem)
            if item:
                x, y, w, h = self.resize_rect(event, item)
                self.memory_rect_pts = [x, y, x + w, y + h]
                self.update()

        # update rectangle position after creating new instance
        if event.type() == QEvent.MouseMove and self.create_instance_flag and self.follow_mouse_flag:
            if isinstance(self.memory_item, TardigradeItem):
                rect = self.resize_rect(event, self.memory_item.get_rectangle())
                self.update()
                self.update_keypoints(self.memory_item.get_keypoints(), rect)
            elif isinstance(self.memory_item, QGraphicsRectItem):
                x, y, w, h = self.resize_rect(event, self.memory_item)
                self.memory_rect_pts = [x, y, x + w, y + h]
                self.update()

        # update label text
        if event.type() == QEvent.MouseButtonDblClick and not self.follow_mouse_flag:
            item = self.check_for_item(QGraphicsTextItem)
            if item:
                text = item.toPlainText()
                cat_list = list(self.category_names.values())
                cat_num = cat_list.index(text)
                new_text = cat_list[cat_num + 1] if cat_num + 1 < len(cat_list) else cat_list[0]
                item.setPlainText(new_text)
                self.update()

        if event.type() == QEvent.MouseButtonPress and self.create_instance_flag:
            if self.follow_mouse_flag:
                # set new instance rectangle second point position
                self.pixmap.setCursor(Qt.ArrowCursor)
                self.follow_mouse_flag = False
                self.create_instance_flag = False
                self.memory_item = None
            else:
                # set new instance rectangle first point position
                class_name = self.category_names[self.window.comboBox.currentText()]
                rect_item = self.create_object_instance(class_name, event)
                self.memory_rect_pts = self.get_position(rect_item)
                self.follow_mouse_flag = True
            self.update()

        # set rectangle to be resized by the mouse movement
        if event.type() == QEvent.MouseButtonDblClick and source == self.graphicsView.viewport():
            self.follow_mouse_flag = True
            item = self.check_for_item(QGraphicsRectItem)
            if item:
                self.memory_rect_pts = self.get_position(item)
            self.update()

        # set rectangle position after double click
        if event.type() == QEvent.MouseButtonPress and self.follow_mouse_flag and not self.create_instance_flag:
            self.follow_mouse_flag = False
            item = self.check_for_item(QGraphicsRectItem)
            if item:
                self.memory_rect_pts = self.get_position(item)
            self.update()

        # resize scene view
        if event.type() == QEvent.Wheel and source == self.graphicsView.viewport() and \
                event.modifiers() == Qt.ControlModifier:
            scale = 1.20 if event.angleDelta().y() > 0 else 0.8
            self.graphicsView.scale(scale, scale)
            self.update()
            return True

        return super().eventFilter(source, event)

    def create_instance(self):
        """Connected to 'Create instance' button - opens QDialog window for the object type selection"""
        self.window = InstanceWindow()
        self.window.show()
        if self.window.exec():
            self.create_instance_flag = True
            self.pixmap.setCursor(Qt.CrossCursor)

    def create_tardigrade(self, class_name, bbox, kpts, value):
        """
        Create tardigrade widget and child items
        :param class_name: str
        :param bbox: tuple - (x1, y1, x2, y2) floating point bbox position
        :param kpts: list - list of lists with [x, y] floating point keypoints positions
        :param value: int - tardigrade number
        :return: QGraphicsRectItem - tardigrade bbox
        """
        colour = self.random_qt_colour()
        rect_item, label = self.add_rect(bbox, class_name, colour=colour)
        points = [self.add_pt(pt, i, colour=colour) for i, pt in enumerate(kpts)]
        tard_item = TardigradeItem(label, rect_item, points)
        tard_item.setData(0, {"label": class_name,
                              "value": value})
        self.scene.addItem(tard_item)
        self.memory_item = tard_item
        return rect_item

    # TODO: Make it possible to change the scale value
    def create_scale(self, bbox):
        """
        Create scale bbox and label items
        :param bbox: tuple - (x1, y1, x2, y2) floating point bbox position
        :return: QGraphicsRectItem - scale bbox
        """
        colour = self.random_qt_colour()
        rect_item, _ = self.add_rect(bbox, "scale", data=self.image_data["scale_value"], colour=colour)
        self.memory_item = rect_item
        return rect_item

    def create_object_instance(self, class_name, event):
        """
        Creates new object instance based on the class_name
        :param class_name: str
        :param event: QEvent
        :return: QGraphicsRectItem
        """
        mouse_x, mouse_y = self.get_mouse_pos(event)
        bbox = [mouse_x, mouse_y, mouse_x + 20, mouse_y + 20]

        if class_name == "scale":
            return self.create_scale(bbox)

        self.tard_num += 1
        kpts = [[mouse_x, mouse_y] for _ in range(7)]
        return self.create_tardigrade(class_name, bbox, kpts, self.tard_num)

    def save_points(self):
        """Save all item changes to the json file corresponding to the current image"""
        img_path = self.images_paths[self.current_image]
        data_path = img_path.parent / (img_path.stem + ".json")
        img_data = {"path": str(img_path)}
        category_list = list(self.category_names.values())
        annotations = []

        for item in self.scene.items():
            if isinstance(item, TardigradeItem):
                annotations.append({
                    "label": category_list.index(item.get_label().toPlainText()) + 1,
                    "bbox": self.get_position(item.get_rectangle()),
                    "keypoints": [self.get_position(pt)[:2] for pt in item.get_keypoints()]})

            elif item.data(0) is not None and item.data(0)["label"] == "scale":
                img_data["scale_value"] = item.data(0)["value"]
                img_data["scale_bbox"] = self.get_position(item)

        img_data["annotations"] = annotations
        ujson.dump(img_data, data_path.open("w"))
        self.msg_queue.put(f"Changes saved to the file: {str(data_path)}.\n")

    def create_items_group(self, position, data, label_txt, item_type="rect", pen_colour=Qt.blue,
                           brush_colour=None):
        """
        Create items groups (ellipses and labels) when item_type is 'point' and
        rectangle with label as a child for item_type 'rect'
        :param position: tuple - (x1, y1, x2, y2) or (x, y) floating point position of the bbox or point respectively
        :param data: int/float - item data (scale value for rectangle corresponding to the scale)
        :param label_txt: str - text for the label to be shown
        :param item_type: str - 'rect' / 'point'
        :param pen_colour: QtGui.QColor - border colour of the item
        :param brush_colour: QtGui.QColor - interior color of the item
        :return: QGraphicsItemGroup/ tuple - items group for the keypoints and tuple with QGraphicsRectItem and
        QGraphicsTextItem for the rectangle
        """
        def move_select_cursor(item):
            item.setFlag(QGraphicsItem.ItemIsMovable)
            item.setFlag(QGraphicsItem.ItemIsSelectable)
            item.setCursor(Qt.CrossCursor)

        def item_settings():
            elem.setData(0, {"label": label_txt,
                             "value": data})
            elem.setPen(QtGui.QPen(Qt.transparent if pen_colour is None else pen_colour))
            elem.setBrush(QtGui.QBrush(Qt.transparent if brush_colour is None else brush_colour))
            move_select_cursor(elem)

        label = QGraphicsTextItem()
        label.setPlainText(label_txt)
        label.setDefaultTextColor(pen_colour)
        label.setPos(position[0], position[1])
        move_select_cursor(label)

        if item_type == "rect":
            x1, y1, x2, y2 = position
            elem = QGraphicsRectItem(x1, y1, abs(x2 - x1), abs(y2 - y1))
            item_settings()
            label.setParentItem(elem)
            return elem, label

        else:
            item = QGraphicsItemGroup()
            elem = QGraphicsEllipseItem(position[0], position[1], self.point_size, self.point_size)
            item_settings()
            [item.addToGroup(it) for it in (elem, label)]
            move_select_cursor(item)
            return item

    def add_rect(self, bbox, class_name, data=None, colour=Qt.blue):
        """Create and add a rectangle to the scene"""
        rect_item, label = self.create_items_group(bbox, data, class_name, item_type="rect", pen_colour=colour)
        self.scene.addItem(rect_item)  # adds all children to the scene automatically (label)
        return rect_item, label

    def add_pt(self, pt, label_num, data=None, colour=Qt.blue):
        """Create and add a point to the scene"""
        pt_item = self.create_items_group(pt, data, f"{label_num + 1}", item_type="point",
                                          pen_colour=colour, brush_colour=colour)
        self.scene.addItem(pt_item)
        return pt_item

    @staticmethod
    def random_qt_colour():
        """
        Generate random RGB colour for the scene object
        :return: QtGui.QColor
        """
        colour = QtGui.QColor()
        colour.setRgb(*[randint(50, 255) for _ in range(3)])
        return colour

    def create_img_objects(self, img_path):
        """
        Download json file associated with passed image path and create and draw file objects
        :param img_path: Path
        """
        data_path = img_path.parent / (img_path.stem + ".json")
        if data_path.is_file():
            random.seed(self.current_image)
            self.image_data = ujson.load(data_path.open("rb"))
            self.create_scale(self.image_data["scale_bbox"])
            category_list = list(self.category_names.values())
            for i, annot in enumerate(self.image_data["annotations"]):
                self.create_tardigrade(category_list[annot["label"] - 1], annot["bbox"], annot["keypoints"], i)
                self.tard_num += 1

    def get_position(self, obj):
        """
        Get passed item position relative to the scene
        :param obj: QGraphicsRectItem/ QGraphicsItemGroup - object whose position is obtained
        :return: list with floating point object coordinates
        """
        dx, dy = 0, 0
        if isinstance(obj, QGraphicsItemGroup):
            # shift position by the ellipse radius
            dx -= self.point_size / 2
            dy -= self.point_size / 2
        x1, y1, x2, y2 = obj.shape().controlPointRect().getCoords()
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
