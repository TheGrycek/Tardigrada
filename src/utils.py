import cv2
import numpy as np


def resize_pad(image, img_size=None, new_size=227, get_scales_only=False):
    if image is not None:
        img_size = image.shape[:2]

    ratio = float(new_size) / max(img_size)
    new_shape = [int(im * ratio) for im in img_size]

    pad_x = new_size - new_shape[1]
    pad_y = new_size - new_shape[0]
    left, top = pad_x // 2, pad_y // 2
    right, bottom = pad_x - (pad_x // 2), pad_y - (pad_y // 2)

    if get_scales_only:
        return ratio, (left, right, top, bottom)

    resized_img = cv2.resize(image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)

    return cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def calculate_mass(points, scale=1, echiniscus=False):
    # If the mass of the species Echiniscus is estimated, use different equation
    # TODO: pass nn input image scale factors and multiply with length and width
    head, ass, right, left = points
    length = np.linalg.norm(head - ass) * scale
    width = np.linalg.norm(right - left) * scale
    R = length / width
    density = 1.04

    if echiniscus:
        mass = (1 / 12) * length * np.pi * (length / R) ** 0.5 * density * 10 ** -6  # [ug]
    else:
        mass = length * np.pi * (length / 2 * R) ** 0.5 * density * 10 ** -6  # [ug]

    print(f"length / width: {R}, mass: {mass}")
    return mass
