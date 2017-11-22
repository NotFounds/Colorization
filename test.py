import os
import argparse
import datetime
import numpy as np
import chainer
import chainer.links as L
from chainer import serializers
from PIL import Image
import model as M
import util

def main():
    print("Chainer Version: " + chainer.__version__)

    parser = argparse.ArgumentParser(description='Colorization')
    parser.add_argument('--dataset', '-d', default='./test')
    parser.add_argument('--model_class', default='./class.model')
    parser.add_argument('--model_color', default='./color.model')
    parser.add_argument('--out', '-o', default='./output', help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (native value indicates CPU)')
    args = parser.parse_args()

    # Set up a neural network
    model_class = L.Classifier(M.Classification(3))
    model_color = M.Evalution(M.Colorization(3))

    # Setup for GPU
    xp = np
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model_class.to_gpu()
        model_color.to_gpu()
        xp = chainer.cuda.cupy

    # Load model
    serializers.load_npz(args.model_class, model_class)
    serializers.load_npz(args.model_color, model_color)

    # Load the dataset
    test_data = args.dataset + '/'
    test, filenames = util.make_testdata(test_data, xp.float32)

    # Make Directory
    output_dir = args.out
    util.make_dir(output_dir)
    print('# Output:    {output_dir}'.format(**locals()))

    # Output test image
    chainer.using_config('train', False)
    for x, name in zip(test, filenames):
        y1 = chainer.functions.softmax(model_class.predictor(xp.asarray([x])))
        y2 = model_color.predictor(xp.asarray([x]))
        if args.gpu >= 0:
            img = util.output2img(chainer.cuda.to_cpu(y2.data[0]))
        else:
            img = util.output2img(y2.data[0])
        img.save('{output_dir}/{name}.png'.format(**locals()))
        l = np.argmax(y1.data)
        m = round(float(np.max(y1.data) * 100), 4)
        print('label: {l}  {m}% \t{name}.png'.format(**locals()))

if __name__ == '__main__':
    main()
