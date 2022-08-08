import numpy as np
from easyocr import Reader
import cv2


def change_to_um(text):
    print(f"MY TEXT: {text}")

    scales = {"um": 1,
              "mm": 0.001,
              "cm": 0.0001}

    text = text.strip()
    text = "".join(text.split())
    text = text.replace(",", ".")

    value, unit = float(text[:-2]), text[-2:]
    value /= scales[unit]

    return value


def read_scale(img, bboxes, device="cpu"):
    if len(bboxes) == 0:
        return {"pix": 0, "um": 0}, img

    reader = Reader(['en'], gpu=True if device == "gpu" else False)

    # TODO: create bboxes filtering algorithm
    x, y, w, h = bboxes[0]
    result = reader.readtext(img,
                             allowlist=["u", "m", "c"] + [str(i) for i in range(10)],
                             slope_ths=5,
                             width_ths=5)

    text_threshold = 0.8
    scale_center = np.array([round(x + w/2), round(y + h/2)])

    def calc_dist(res):
        p1, p2, p3, p4 = res[0]
        centers = np.array([abs(p1[0] - p2[0]) / 2 + p1[0],
                            abs(p1[1] - p4[1]) / 2 + p1[1]])

        return np.power(np.sum((scale_center - centers) ** 2), 0.5)

    result = filter(lambda res: res[1][-1] == "m", result)
    result = filter(lambda res: res[2] >= text_threshold, result)
    result = sorted(result, key=calc_dist, reverse=True)
    print(f"SCALE RESULT: {result}")

    if len(result) == 0:
        return {"pix": 0, "um": 0}, img

    scale_value = change_to_um(result[0][1])

    img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)
    img = cv2.putText(img, 'scale', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (255, 0, 0), 2, cv2.LINE_AA)

    output = {"pix": w, "um": scale_value}
    return output, img
