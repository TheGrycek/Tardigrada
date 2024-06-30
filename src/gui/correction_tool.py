import random
from collections import defaultdict
from random import randint

import numpy as np
import seaborn as sns
import ujson
from PyQt5 import QtGui
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import (QMainWindow, QGraphicsScene, QGraphicsPixmapItem,
                             QGraphicsItem, QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsRectItem,
                             QGraphicsTextItem)
from gui.utils import TardigradeItem, InstanceWindow


class CorrectionTool(QMainWindow):
    """Application functionality base class"""
    def __init__(self):
        super().__init__()
        self.initialize_scene()
        self.image_data = defaultdict(lambda: None)
        self.first_image = True
        self.current_image = 0
        self.images_paths = []
        self.follow_mouse_flag = False
        self.create_instance_flag = False
        self.tard_num = 0
        self.memory_item = None
        self.memory_rect_pts = None
        self.point_size = 3
        self.pen_size = 2
        self.palette = (np.array(sns.color_palette('gist_rainbow', n_colors=1000)) * 255).astype(int)

    def initialize_scene(self):
        self.scene = QGraphicsScene()
        self.pixmap = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap)

    def set_scene(self, img_path):
        """
        Add an image to the 'Correction Tool' scene and draw object associated with the passed image path
        :param img_path: Path - path from which the images are downloaded
        """
        self.initialize_scene()
        self.create_img_objects(img_path)
        self.pixmap.setPixmap(QtGui.QPixmap(str(img_path)))
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        if len(self.image_data):
            scale_val = self.image_data["scale_value"]
            if scale_val is not None:
                self.scaleSpinBox.setValue(scale_val)

    def check_for_item(self, item_class):
        """Finds an object of the specified type in the list of selected scene items"""
        items = self.scene.selectedItems()
        if len(items) > 0:
            item = items[0]
            if item_class is not None:
                if isinstance(item, item_class):
                    return item
            else:
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
        x1, y1, x2, y2 = self.memory_rect_pts

        pos_delta = item.pos()
        dx, dy = pos_delta.x(), pos_delta.y()

        x, y = x1, y1
        w = abs(x - (mouse_x - dx))
        h = abs(y - (mouse_y - dy))

        # resize rectangle in every direction
        if mouse_x > x1 and mouse_y < y1:
            x, y = x1, mouse_y
        elif mouse_x < x1 and mouse_y < y1:
            x, y = mouse_x, mouse_y
        elif mouse_x < x1 and mouse_y > y1:
            x, y = mouse_x, y1

        item.setRect(x, y, w, h)
        self.update()
        return [x, y, w, h]

    def stack_widget_on_top(self, item):
        """
        Put the tardigrade object on top of the other to easily manipulate it
        :param item: QGraphicItem - object on the scene
        """
        if item:
            max_z_val = 0
            for scene_item in self.scene.items():
                item_z_val = scene_item.zValue()
                if item_z_val > max_z_val:
                    max_z_val = item_z_val

            item.set_z_value(max_z_val + 1)

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

        if event.type() == QEvent.MouseButtonRelease:
            item = self.check_for_item(None)
            if item:
                self.stack_widget_on_top(item.data(1))
                self.update()

        # update rectangle position after double click
        if event.type() == QEvent.MouseMove and self.follow_mouse_flag:
            item = self.check_for_item(QGraphicsRectItem)
            if item and self.memory_rect_pts:
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
                self.memory_rect_pts = rect_item.rect().getCoords()
                self.follow_mouse_flag = True
            self.update()

        # set rectangle to be resized by the mouse movement
        if event.type() == QEvent.MouseButtonDblClick and source == self.graphicsView.viewport():
            self.follow_mouse_flag = True
            item = self.check_for_item(QGraphicsRectItem)
            if item:
                self.memory_rect_pts = item.rect().getCoords()
                self.update()

        # set rectangle position after double click
        if event.type() == QEvent.MouseButtonPress and self.follow_mouse_flag and not self.create_instance_flag:
            self.follow_mouse_flag = False
            item = self.check_for_item(QGraphicsRectItem)
            if item:
                self.memory_rect_pts = item.rect().getCoords()
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
        self.window = InstanceWindow(str(self.find_file_dir("src/gui/instance_window.ui")))
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

    def scale_value_change(self, value):
        for item in self.scene.items():
            if item.data(0) is not None and item.data(0)["label"] == "scale":
                item.data(0)["value"] = value
                item.setData(0, {"label": "scale", "value": value})
                self.image_data["scale_value"] = value

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
                    "bbox": self.get_shifted_position(item.get_rectangle()),
                    "keypoints": [self.get_shifted_position(pt) for pt in item.get_keypoints()]})

            elif item.data(0) is not None and item.data(0)["label"] == "scale":
                img_data["scale_value"] = item.data(0)["value"]
                img_data["scale_bbox"] = self.get_shifted_position(item)

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

        def elem_settings():
            elem.setData(0, {"label": label_txt,
                             "value": data})
            elem.setPen(pen)
            elem.setBrush(brush)
            move_select_cursor(elem)

        label = QGraphicsTextItem()
        label.setPlainText(label_txt)
        label.setDefaultTextColor(pen_colour)
        label.setPos(position[0], position[1])
        move_select_cursor(label)

        pen = QtGui.QPen(Qt.transparent if pen_colour is None else pen_colour)
        pen.setWidth(self.pen_size)
        brush = QtGui.QBrush(Qt.transparent if brush_colour is None else brush_colour)

        if item_type == "rect":
            x1, y1, x2, y2 = position
            elem = QGraphicsRectItem(x1, y1, abs(x2 - x1), abs(y2 - y1))
            elem_settings()
            label.setParentItem(elem)
            return elem, label

        else:
            item = QGraphicsItemGroup()
            radius = self.point_size / 2
            x = position[0] - radius
            y = position[1] - radius

            elem = QGraphicsEllipseItem(x, y, self.point_size, self.point_size)
            elem_settings()
            [item.addToGroup(el) for el in (elem, label)]
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

    def random_qt_colour(self):
        """
        Generate random RGB colour for the scene object
        :return: QtGui.QColor
        """
        colour = QtGui.QColor()
        colour.setRgb(*self.palette[randint(0, len(self.palette))])
        return colour

    def create_img_objects(self, img_path):
        """
        Download json file associated with passed image path and create and draw file objects
        :param img_path: Path
        """
        data_path = img_path.parent / (img_path.stem + ".json")
        random.seed(self.current_image)
        try:
            self.image_data = ujson.load(data_path.open("rb"))
            self.create_scale(self.image_data["scale_bbox"])
            category_list = list(self.category_names.values())
            for i, annot in enumerate(self.image_data["annotations"]):
                self.create_tardigrade(category_list[annot["label"] - 1], annot["bbox"], annot["keypoints"], i)
                self.tard_num += 1
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            self.image_data = {"path": data_path, "scale_value": 0.0, "scale_box": [], "annotations": []}
            self.msg_queue.put(f"Loading labels from file {data_path} failed.\n")

    def get_shifted_position(self, obj):
        """
        Get passed item position relative to the scene
        :param obj: QGraphicsRectItem/ QGraphicsItemGroup - object whose position is obtained
        :return: list with floating point object coordinates
        """
        pos_d = obj.pos()
        dx, dy = pos_d.x(), pos_d.y()

        if isinstance(obj, QGraphicsRectItem):
            x1, y1, x2, y2 = obj.rect().getCoords()
            return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

        elif isinstance(obj, QGraphicsItemGroup):
            ellipse = obj.childItems()[0]
            x1, y1, _, _ = ellipse.rect().getRect()
            radius = self.point_size / 2
            return [x1 + dx + radius, y1 + dy + radius]

    def closeEvent(self, event):
        if self.auto_save:
            self.save_points()

        event.accept()
