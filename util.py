import os
import os.path
import numpy as np
import chainer
from chainer import datasets
from PIL import Image, ImageFilter

def output2img(y):
    return Image.fromarray(np.asarray((np.transpose(y, (1, 2, 0))) * 255, dtype=np.uint8))

def output2img_hsv(x, y):
    hsv = np.asarray([y[0][0], y[0][1], x[0]])
    return Image.fromarray(np.asarray((np.transpose(hsv, (1, 2, 0))) * 255.0, dtype=np.uint8)).convert('RGB')

def rgb2hsv(img, dtype=np.float32):
    hsv = np.asarray(img.convert('HSV'), dtype)
    return np.transpose(hsv, (2, 0, 1))/255.0

def make_dataset(dir, dtype=np.float32):
    paths = os.listdir(dir)
    input_imgs = []
    truth_imgs = []
    for path in paths:
        # Open Image
        img = Image.open(dir + '/' + path)

        # Truth(Color) Image
        r, g, b, a = img.convert('RGBA').split()
        r_img = np.asarray(dtype(r)/255.0)
        g_img = np.asarray(dtype(g)/255.0)
        b_img = np.asarray(dtype(b)/255.0)
        truth_imgs.append(np.asarray([r_img, g_img, b_img], dtype=dtype))

        # Gray Image
        l = np.asarray(dtype(img.convert('L'))/255.0)

        # Edge Image
        filter_img = laplacian(img)
        f = np.asarray(dtype(filter_img.convert('L'))/255.0)

        # Create Input data
        input_imgs.append(np.asarray([l, l, l, f], dtype=dtype))

    return datasets.TupleDataset(input_imgs, truth_imgs)

def laplacian(img):
    # Laplacian filter (second order differential)
    flt = ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1)
    return img.filter(flt)

def make_testdata(dir, dtype=np.float32):
    paths = os.listdir(dir)
    names = []
    input_imgs = []
    for path in paths:
        # Open Image
        img = Image.open(dir + '/' + path)

        # Gray Image
        l = np.asarray(dtype(img.convert('L'))/255.0)

        # Edge Image
        filter_img = laplacian(img)
        f = np.asarray(dtype(filter_img.convert('L'))/255.0)

        # Create Input data
        input_imgs.append(np.asarray([l, l, l, f], dtype=dtype))

        filename = os.path.splitext(os.path.basename(path))[0]
        names.append(filename)

    return input_imgs, names

def next_dir(dir):
    i = 1
    while os.path.isdir(dir + str(i)):
        i += 1
    return dir + str(i)

def make_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
