import cv2


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
