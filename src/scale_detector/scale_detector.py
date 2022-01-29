from easyocr import Reader
import cv2


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


def read_scale(img, rect, device="cpu"):
    reader = Reader(['en'], gpu=True if device == "gpu" else False)

    x, y, w, h = rect
    result = reader.readtext(img,
                             slope_ths=0.01,
                             width_ths=0.7)

    text_threshold = 0.8
    scale_center = (round(x + w/2), round(y + h/2))

    def calc_dist(res):
        p1, p2, p3, p4 = res[0]
        center_x = abs(p1[0] - p2[0]) / 2 + p1[0]
        center_y = abs(p1[1] - p4[1]) / 2 + p1[1]
        return (abs(scale_center[0] - center_x) ** 2 + abs(scale_center[1] - center_y) ** 2) ** 0.5

    result = filter(lambda res: res[2] > text_threshold, result)
    result = sorted(result, key=calc_dist, reverse=True)[0]
    scale_value = change_to_um(result[1])

    img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)
    img = cv2.putText(img, 'scale', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (255, 0, 0), 2, cv2.LINE_AA)

    output = {"pix": w, "um": scale_value}
    return output, img
