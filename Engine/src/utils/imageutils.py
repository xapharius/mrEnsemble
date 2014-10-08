__author__ = 'simon'

import os
import numpy as np
import numpyutils as nputils
from skimage import img_as_float
import skimage.io as io
from skimage.transform import resize


def load_image(img_file, normalize=False, scale_to=None):
    img = img_as_float(io.imread(img_file, as_grey=True))
    if scale_to is not None:
        img = resize(img, scale_to)
    if normalize:
        img = nputils.normalize_image(img, -1, 1)
    return img


def load_images(path, max_num=-1, normalize=False, scale_to=None):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.png'):
            img = load_image(path + '/' + _file, normalize=normalize, scale_to=scale_to)
            images.append(img)
            if len(images) == max_num:
                break
    return images


def load_image_and_split(path, single_size):
    images = []
    img = img_as_float(io.imread(path, as_grey=True))
    img_shape = np.shape(img)
    for y in range(0, img_shape[0], single_size[0]):
        for x in range(0, img_shape[1], single_size[1]):
            images.append(np.copy(img[y:y+single_size[0], x:x+single_size[1]]))
    return images