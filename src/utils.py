import cv2
import numpy as np
from key_points_detector.dataset import resize_pad


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


def prepare_segmented_img(img, cnt_translated, shape_translated, bbox):
    mask = np.zeros((shape_translated[1][1] + 1, shape_translated[1][0] + 1, 3), dtype=np.uint8)
    cv2.drawContours(mask, [np.array(cnt_translated)], -1, (255, 255, 255), thickness=cv2.FILLED)

    x, y, w, h, = bbox
    result_img = cv2.bitwise_and(mask, img[y: y + h, x: x + w])

    return result_img


def prepare_contours(rect_tilted, rect, cnt):
    center_unscaled, size, angle = rect_tilted[0], rect_tilted[1], rect_tilted[2]
    cnt_normalized, cnt_translated, center, scale, shape_translated = normalize_points(rect, cnt, center_unscaled)

    return cnt_normalized, cnt_translated, center, shape_translated
