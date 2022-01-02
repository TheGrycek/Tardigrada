import argparse
import json
from pathlib import Path

from scale_detector.scale_detector import read_scale  # needs to be before import cv2, to avoid SIGSEGV
import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import resize_pad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, default="./images/krio5_OM_1.5_1.jpg",
                        help="Input image directory.")

    return parser.parse_args()


def normalize_points(bbox, points, center):
    new_size = 227
    ratio, (left, right, top, bottom) = resize_pad(None, get_scales_only=True, img_size=bbox[-2:],
                                                   new_size=new_size)
    x = points[:, :, 0].flatten()
    y = points[:, :, 1].flatten()

    max_x, min_x = np.max(x), np.min(x)
    max_y, min_y = np.max(y), np.min(y)

    center_x = (center[0] - min_x) / (max_x - min_x)
    center_y = (center[1] - min_y) / (max_y - min_y)

    scale_x, scale_y = (abs(max_x - min_x), abs(max_y - min_y))

    points_resized = points.astype(np.float64)
    points_resized[:, :, 0] -= min_x
    points_resized[:, :, 1] -= min_y
    points_resized *= ratio
    points_resized = points_resized.astype(np.int32)
    points_resized[:, :, 0] += top
    points_resized[:, :, 1] += left

    shape_translated = [[0, 0], [max_x - min_x, max_y - min_y]]

    new_points = []
    new_points_translated = []
    for pt, pt_r, in zip(points, points_resized):
        x = pt_r[0][0] / new_size
        y = pt_r[0][1] / new_size
        xt, yt = pt[0][0] - min_x, pt[0][1] - min_y

        new_points_translated.append([xt, yt])
        new_points.append([x, y])

    new_points.append(new_points[0])  # CLOSE CONTOUR
    # print(f"SCALE X: {scale_x}, SCALE Y: {scale_y}")

    return new_points, new_points_translated, (center_x, center_y), (scale_x, scale_y), shape_translated


def prepare_contours(rect_tilted, rect, cnt):
    center_unscaled, size, angle = rect_tilted[0], rect_tilted[1], rect_tilted[2]
    cnt_normalized, cnt_translated, center, scale, shape_translated = normalize_points(rect, cnt, center_unscaled)

    return cnt_normalized, cnt_translated, center, shape_translated


def save_image(image):
    global FILE_NUM
    file_path = Path(f"images/dataset/label{FILE_NUM}.png")
    cv2.imwrite(str(file_path), image)


def save_label(points, labels):
    global FILE_NUM
    out = {"contour_points": list(points), "key_points": labels}
    out_file = open(f"images/dataset/label{FILE_NUM}.json", "w")
    json.dump(out, out_file, indent=6)
    out_file.close()
    FILE_NUM += 1


def prepare_segmented_img(img, cnt_translated, shape_translated, bbox):
    mask = np.zeros((shape_translated[1][1] + 1, shape_translated[1][0] + 1, 3), dtype=np.uint8)
    cv2.drawContours(mask, [np.array(cnt_translated)], -1, (255, 255, 255), thickness=cv2.FILLED)

    x, y, w, h, = bbox
    result_img = cv2.bitwise_and(mask, img[y: y + h, x: x + w])

    return result_img


def label_img(img, pts_normalized, cnt_translated, center, shape_translated, bbox):
    xr = np.array(pts_normalized)[:, 0].flatten()
    yr = np.array(pts_normalized)[:, 1].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    col = np.arange(len(xr))

    ax.scatter(*center, color="red")
    ax.plot(xr, yr, color="red", picker=3)
    ax.scatter(xr, yr, s=10, c=col, marker='o')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    labels = []
    labels_points = []

    def on_pick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        x, y = np.take(xdata, ind)[0], np.take(ydata, ind)[0]
        print(f"Selected point: {x, y}")

        if len(labels) <= 4:
            labels.append([x, y])
            if len(labels) == 4:
                print("Four points selected!\n")
        elif len(labels) > 4:
            print(f"Too many label points: {len(labels)}!\n")

        index = 0
        old_len = 1
        for i, p in enumerate(pts_normalized):
            length = (abs(x - p[0]) ** 2 + abs(y - p[1]) ** 2) ** 0.5
            if length < old_len:
                index = i
                old_len = length

        labels_points.append(index)

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

    label_image = prepare_segmented_img(img, cnt_translated, shape_translated, bbox)
    save_image(label_image)
    save_label(pts_normalized, labels)


def main(args):
    img = cv2.imread(str(args.input_dir), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnt, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ind = 0
    for c in cnt[1:]:
        if cv2.contourArea(c) > 1000:
            rect_tilted = cv2.minAreaRect(c)

            if 0.5 < rect_tilted[1][0] / rect_tilted[1][1] < 2:
                continue

            rect = cv2.boundingRect(c)
            x, y, w, h = rect

            if ind == 0:
                image_scale = read_scale(img, rect)
                print(f"Image scale: {image_scale}")

            cnt_normalized, cnt_translated, center, shape_translated = prepare_contours(rect_tilted, rect, c)
            label_img(img, cnt_normalized, cnt_translated, center, shape_translated, rect)

            box = np.int0(cv2.boxPoints(rect_tilted))
            cv2.drawContours(img, [box], 0, (0, 0, 255))
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(f'./crops/tard_{ind}.png', cv2.resize(thresh[y: y + h, x: x + w], (0, 0), fx=2, fy=2))
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

            ind += 1

    cv2.imshow('oryginal', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
    cv2.imshow('threshold', cv2.resize(thresh, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(0)


if __name__ == '__main__':
    FILE_NUM = 0
    main(parse_args())
