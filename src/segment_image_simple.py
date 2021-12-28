import argparse
from pathlib import Path

import cv2
import numpy as np
from scale_detector.scale_detector import read_scale


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, default="./images/krio5_OM_2_6.jpg",
                        help="Input image directory.")

    return parser.parse_args()


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def crop_rect(img, rect_tilted, rect, cnt):
    center, size, angle = rect_tilted[0], rect_tilted[1], rect_tilted[2]
    small_center = [size[0]/2, size[1]/2]
    translation = [center[0] - small_center[0], center[1] - small_center[1]]
    # print(f"SIZE: {size}, CENTER: {center}, TRAANSLATION {translation}")
    center, size = tuple(map(round, center)), tuple(map(round, size))
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    moments = cv2.moments(cnt)
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas - angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)
    cnt_transated = np.zeros_like(cnt_rotated)

    for i, c in enumerate(cnt_rotated):
        cnt_transated[i] = (c - translation).astype(np.int32)

    img_rot = cv2.cvtColor(img_rot, cv2.COLOR_GRAY2BGR)
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(img_rot, [cnt_rotated], -1, (0, 255, 0), 2)
    cv2.drawContours(img_crop, cnt_transated, -1, (0, 255, 0), 2)
    img_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

    return img_rot, img_crop, cnt_rotated


def main(args):
    img_original = cv2.imread(str(args.input_dir), 1)
    img = img_original.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnt, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for c in cnt:
        if cv2.contourArea(c) > 1000:
            rect_tilted = cv2.minAreaRect(c)

            if 0.5 < rect_tilted[1][0] / rect_tilted[1][1] < 2:
                continue
            box = np.int0(cv2.boxPoints(rect_tilted))
            cv2.drawContours(img, [box], 0, (0, 0, 255))

            rect = cv2.boundingRect(c)
            x, y, w, h = rect

            if i == 0:
                image_scale = read_scale(img_original, rect)
                print(image_scale)

            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img_rot, img_crop, new_c = crop_rect(thresh, rect_tilted, rect, c)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            epsilon = 0.008 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            approx_rot = cv2.approxPolyDP(new_c, epsilon, True)

            cv2.drawContours(img, [approx_rot], -1, (255, 0, 0), 2)
            for app in approx:
                cv2.circle(thresh, tuple(app[0]), 2, [255, 0, 0], -1)

            cv2.imwrite(f'./crops/tard_{i}.png', cv2.resize(thresh[y: y + h, x: x + w], (0, 0), fx=2, fy=2))
            i += 1
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    cv2.imshow('oryginal', cv2.resize(img, (0, 0), fx=0.3, fy=0.3))
    cv2.imshow('threshold', cv2.resize(thresh, (0, 0), fx=0.3, fy=0.3))
    cv2.waitKey(0)


if __name__ == '__main__':
    main(parse_args())
