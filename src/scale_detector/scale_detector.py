import cv2
import numpy as np
from easyocr import Reader


def change_to_um(text):
    scales = {"um": 1,
              "mm": 0.001,
              "cm": 0.0001}

    text = text.strip()
    text = "".join(text.split())
    text = text.replace(",", ".")

    value, unit = float(text[:-2]), text[-2:]
    value /= scales[unit]

    return value


def calc_overlap_perc(bbox1, bbox2):
    x11, x12, y11, y12 = bbox1
    x21, x22, y21, y22 = bbox2
    s1 = (abs(x11 - x12) * abs(y11 - y12))
    si = max(0, min(x12, x22) - max(x11, x21)) * max(0, min(y12, y22) - max(y11, y21))
    su = s1 - si
    return 1 if su == 0 else si / su


def filter_cnts(cnts, scale_pts):
    cnts_filtered = []
    bboxes_fit = []
    bboxes = []
    p1, p2, p3, p4, = scale_pts
    l, h = abs(p1[0] - p2[0]), abs(p1[1] - p4[1])
    text_len = max(l, h)

    for cnt in cnts:
        rect = cv2.boundingRect(cnt)
        bbox_len = rect[2] if text_len == l else rect[3]
        if bbox_len < text_len:
            continue
        overlap_perc = calc_overlap_perc([p1[0], p2[0], p1[1], p4[1]],
                                         [rect[0], rect[0] + rect[2], rect[1], rect[1] + rect[3]])
        if overlap_perc > 0.5:
            continue

        rect_tilted = cv2.minAreaRect(cnt)
        if rect_tilted[1][0] == 0 or rect_tilted[1][1] == 0:
            continue
        if 0.5 < rect_tilted[1][0] / rect_tilted[1][1] < 2:
            continue

        bboxes.append(rect)
        cnts_filtered.append(cnt)
        bboxes_fit.append(rect_tilted)

    return cnts_filtered, bboxes_fit, bboxes


def simple_segmenter(img, scale_pts):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts, bboxes_fit, bboxes = filter_cnts(cnts, scale_pts)

    return cnts, bboxes_fit, bboxes, thresh


def select_bbox(bboxes, scale_pts, img):
    p1, p2, p3, p4 = scale_pts
    text_center = np.array([abs(p1[0] - p2[0]) / 2 + p1[0], abs(p1[1] - p4[1]) / 2 + p1[1]])

    def calc_dist(bbox):
        x, y, w, h = bbox
        bbox_center = np.array([round(x + w / 2), round(y + h / 2)])
        return np.power(np.sum((bbox_center - text_center) ** 2), 0.5)

    bboxes = sorted(bboxes, key=calc_dist, reverse=False)
    for x, y, w, h in bboxes:
        img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)
    return bboxes[0]


def read_scale(img, device="cpu"):
    reader = Reader(['en'], gpu=True if device == "gpu" else False)
    result = reader.readtext(img,
                             allowlist=["u", "m", "c"] + [str(i) for i in range(10)],
                             slope_ths=5,
                             width_ths=5)

    text_threshold = 0.8

    result = filter(lambda res: res[1][-1] == "m", result)
    result = list(filter(lambda res: res[2] >= text_threshold, result))

    if len(result) == 0:
        return {"um": 0, "bbox": []}, img

    cnts, bboxes_fit, bboxes, thresh = simple_segmenter(img, result[0][0])

    scale_value = change_to_um(result[0][1])
    x, y, w, h = select_bbox(bboxes, result[0][0], img)

    img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)
    img = cv2.putText(img, 'scale', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (255, 0, 0), 2, cv2.LINE_AA)

    output = {"um": scale_value, "bbox": [x, y, x + w, y + h]}

    return output, img
