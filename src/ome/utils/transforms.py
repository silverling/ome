import cv2
import numpy as np


def contain_into(img, shape):
    """Contain image into given shape by padding with black pixels."""
    h, w = img.shape[:2]
    target_h, target_w = shape

    scale = min(target_h / h, target_w / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h))

    padded_img = np.zeros((target_h, target_w, *img.shape[2:]), dtype=img.dtype)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded_img[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_img

    return padded_img


def resize_and_center_crop(img, shape):
    """Resize and center crop image to given shape."""
    h, w = img.shape[:2]
    target_h, target_w = shape

    scale = max(target_h / h, target_w / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h))

    start_y = (new_h - target_h) // 2
    start_x = (new_w - target_w) // 2
    cropped_img = resized_img[start_y : start_y + target_h, start_x : start_x + target_w]

    return cropped_img
