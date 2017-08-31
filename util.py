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

def make_dataset(dir, dtype=np.float32):
    paths = os.listdir(dir)
    gray_img = []
    rgb_img = []
    for path in paths:
        img = Image.open(dir + '/' + path)
        r, g, b, a = img.split()
        r_img = np.asarray(dtype(r)/127.5-1)
        g_img = np.asarray(dtype(g)/127.5-1)
        b_img = np.asarray(dtype(b)/127.5-1)
        rgb  = np.asarray([r_img, g_img, b_img])
        rgb_img.append(rgb)
        gray = np.asarray([((77 * dtype(r) + 150 * dtype(g) + 29 * dtype(b)) / 256) / 127.5 - 1])
        gray_img.append(gray)

    threshold = np.int32(len(paths)/10*9)
    train = chainer.datasets.TupleDataset(gray_img[0:threshold], rgb_img[0:threshold])
    test  = chainer.datasets.TupleDataset(gray_img[threshold:],  rgb_img[threshold:])
    return train, test

def make_testdata(dir, dtype=np.float32):
    paths = os.listdir(dir)
    gray_img = []
    for path in paths:
        img = Image.open(dir + '/' + path)
        r, g, b, a = img.split()
        r_img = np.asarray(dtype(r)/127.5-1)
        g_img = np.asarray(dtype(g)/127.5-1)
        b_img = np.asarray(dtype(b)/127.5-1)
        gray = np.asarray([(dtype(r) + dtype(g) + dtype(b)) / 3 / 127.5 - 1])
        gray_img.append(gray)

    test = chainer.datasets.TupleDataset(gray_img,  np.asarray([None] * len(gray_img)))
    return test

def make_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
