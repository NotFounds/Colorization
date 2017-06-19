import os
import os.path
import numpy as np
import chainer
from PIL import Image

def output2img(y, flag=False):
    if flag:
        _max = np.max(y)
        _min = np.min(y)
        return np.asarray((np.transpose(y, (0, 2, 3, 1)) - _min) * (255 / (_max - _min)), dtype=np.uint8)
    else:
        return np.asarray((np.transpose(y, (0, 2, 3, 1)) + 1) * 127.5, dtype=np.uint8)

def read_train_data(input, output, dtype=int):
    return chainer.datasets.TupleDataset(_read_gray_image(input, dtype), _read_rgb_image(output, dtype))

def read_test_data(input, output=None, dtype=int):
    if (output is None):
        return chainer.datasets.TupleDataset(_read_gray_image(input, dtype))
    else:
        return chainer.datasets.TupleDataset(_read_gray_image(input, dtype), _read_rgb_image(output, dtype))

def _read_rgb_image(dir, dtype):
    paths = os.listdir(dir)
    images = []
    for path in paths:
        img = Image.open(dir + path)
        r, g, b, a = img.split()
        r_data = np.asarray(np.float32(r)/255.0)
        g_data = np.asarray(np.float32(g)/255.0)
        b_data = np.asarray(np.float32(b)/255.0)
        img_data = np.asarray([r_data, g_data, b_data])
        images.append(img_data)
    return images

def _read_gray_image(dir, dtype):
    paths = os.listdir(dir)
    images = []
    for path in paths:
        img = Image.open(dir + path)
        r, g, b, a = img.split()
        data = np.asarray(np.float32(r)/255.0)
        img_data = np.asarray([data])
        images.append(img_data)
    return images

def make_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
