import os
import os.path
import numpy as np
import chainer
from PIL import Image

def output2img(y):
    return Image.fromarray(np.asarray((np.transpose(y, (1, 2, 0))) * 255, dtype=np.uint8))

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
        r_img = np.asarray(dtype(r)/255.0)
        g_img = np.asarray(dtype(g)/255.0)
        b_img = np.asarray(dtype(b)/255.0)
        rgb  = np.asarray([r_img, g_img, b_img], dtype=dtype)
        rgb_img.append(rgb)
        v = ((77 * dtype(r) + 150 * dtype(g) + 29 * dtype(b)) / 256) / 255.0

        gray = np.asarray([v, v, v], dtype=dtype)
        gray_img.append(gray)

    threshold = np.int32(len(paths)/10*9)
    train = chainer.datasets.TupleDataset(gray_img[0:threshold], rgb_img[0:threshold])
    test  = chainer.datasets.TupleDataset(gray_img[threshold:],  rgb_img[threshold:])
    return train, test

def make_testdata(dir, dtype=np.float32):
    paths = os.listdir(dir)
    names = []
    gray_img = []
    for path in paths:
        img = Image.open(dir + '/' + path)
        r, g, b, a = img.split()
        r_img = np.asarray(dtype(r)/255.0)
        g_img = np.asarray(dtype(g)/255.0)
        b_img = np.asarray(dtype(b)/255.0)
        v = ((77 * dtype(r) + 150 * dtype(g) + 29 * dtype(b)) / 256) / 255.0

        gray = np.asarray([v, v, v])
        gray_img.append(gray)

        filename = os.path.splitext(os.path.basename(path))[0]
        names.append(filename)

    test = chainer.datasets.TupleDataset(gray_img, gray_img)
    return test, names

def next_dir(dir):
    i = 1
    while os.path.isdir(dir + str(i)):
        i += 1
    return dir + str(i)

def make_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)