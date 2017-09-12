import os
import os.path
import numpy as np
import chainer
from PIL import Image

def output2img(y):
    return Image.fromarray(np.asarray((np.transpose(y, (1, 2, 0)) + 1) * 127.5, dtype=np.uint8))

def output2img_hsv(x, y):
    hsv = np.asarray([y[0][0], y[0][1], x[0]])
    return Image.fromarray(np.asarray((np.transpose(hsv, (1, 2, 0)) + 1) * 127.5, dtype=np.uint8)).convert('RGB')

def rgb2hsv(img, dtype=np.float32):
    hsv = np.asarray(img.convert('HSV'), dtype)
    return np.transpose(hsv, (2, 0, 1))/127.5-1

def make_dataset(dir, dtype=np.float32):
    paths = os.listdir(dir)
    gray_img = []
    rgb_img = []
    for path in paths:
        img = Image.open(dir + '/' + path).convert('RGBA')
        r, g, b, a = img.split()
        r_img = np.asarray(dtype(r)/127.5-1)
        g_img = np.asarray(dtype(g)/127.5-1)
        b_img = np.asarray(dtype(b)/127.5-1)
        rgb  = np.asarray([r_img, g_img, b_img], dtype=dtype)
        rgb_img.append(rgb)
        v = ((77 * dtype(r) + 150 * dtype(g) + 29 * dtype(b)) / 256) / 127.5 - 1
        gray = np.asarray([v, v, v], dtype=dtype)
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
        v = ((77 * dtype(r) + 150 * dtype(g) + 29 * dtype(b)) / 256) / 127.5 - 1
        gray = np.asarray([v, v, v])
        gray_img.append(gray)

    test = chainer.datasets.TupleDataset(gray_img, gray_img)
    return test

def make_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
