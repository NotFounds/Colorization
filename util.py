import os
import os.path
import numpy as np
import chainer
from chainer import datasets
from PIL import Image

def output2img(y):
    return Image.fromarray(np.asarray((np.transpose(y, (1, 2, 0))) * 255, dtype=np.uint8))

def make_dataset(dir, dtype=np.float32):
    input_imgs = []
    labels = []
    truth_imgs = []
    i = 0
    for path in os.listdir(dir):
        if not os.path.isdir(dir + '/' + path):
            continue
        for file in os.listdir(dir + '/' + path):
            # Open image
            img = Image.open(dir + '/' + path + '/' + file)

            # Truth(Color) image
            r, g, b, a = img.convert('RGBA').split()
            r_img = np.asarray(dtype(r) / 255.0)
            g_img = np.asarray(dtype(g) / 255.0)
            b_img = np.asarray(dtype(b) / 255.0)
            truth_img = np.asarray([r_img, g_img, b_img], dtype=dtype)
            truth_imgs.append(truth_img)

            # Gray image
            l = np.asarray(dtype(img.convert('L')) / 255.0)

            # Create input data
            input_imgs.append(np.asarray([l, l, l], dtype=dtype))

            # Add label
            labels.append(i)
        print('{path} : {i}'.format(**locals()))
        i += 1

    return datasets.TupleDataset(input_imgs, labels), datasets.TupleDataset(input_imgs, truth_imgs)

def make_testdata(dir, dtype=np.float32):
    names = []
    input_imgs = []
    if os.path.isdir(dir):
        paths = os.listdir(dir)
        for path in paths:
            # Open image
            img = Image.open(dir + '/' + path).resize((256, 256))

            # Gray image
            l = np.asarray(dtype(img.convert('L')) / 255.0)

            # Create input data
            input_imgs.append(np.asarray([l, l, l], dtype=dtype))
            filename = os.path.splitext(os.path.basename(path))[0]
            names.append(filename)
    else:
        # Open image
        img = Image.open(dir[:-1]).resize((256, 256))

        # Gray image
        l = np.asarray(dtype(img.convert('L')) / 255.0)

        # Create input data
        input_imgs.append(np.asarray([l, l, l], dtype=dtype))
        filename = os.path.splitext(os.path.basename(dir))[0]
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