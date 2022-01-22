import argparse
import json
from pathlib import Path

from scale_detector.scale_detector import read_scale  # needs to be before import cv2, to avoid SIGSEGV
import cv2
import matplotlib.pyplot as plt
import numpy as np

from segmenter.model import simple_segmenter
from utils import prepare_segmented_img, prepare_contours


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, default="./images/krio5_OM_1.5_1.jpg",
                        help="Input image directory.")

    return parser.parse_args()


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
    cnt, bboxes_fit, bboxes, thresh = simple_segmenter(img)

    ind = 0
    for c, rect_tilted, rect in zip(cnt, bboxes_fit, bboxes):
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
