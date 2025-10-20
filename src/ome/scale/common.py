def min_max_scale(img, /):
    img = (img - img.min()) / (img.max() - img.min())
    return img
