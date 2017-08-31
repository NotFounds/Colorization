import argparse
import datetime
import numpy as np
import chainer
from chainer import serializers
from PIL import Image
import model as M
import util

def main():
    print(chainer.__version__)

    parser = argparse.ArgumentParser(description='Colorization')
    parser.add_argument('--dataset', '-d', default='./test')
    parser.add_argument('--model', '-m', default='./example.model')
    parser.add_argument('--out', '-o', default='./output', help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (native value indicates CPU)')
    args = parser.parse_args()

    # Set up a neural network
    model = M.Evalution(M.Colorization(2, 3))

    # Setup for GPU
    xp = np
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = chainer.cuda.cupy
        print('GPU:       {args.gpu}'.format(**locals()))

    # Load model
    serializers.load_npz(args.model, model)

    # Load the dataset
    test_data = args.dataset + '/'
    test = util.make_testdata(test_data, xp.float32)
    test_itr = chainer.iterators.SerialIterator(test, 1, repeat=False, shuffle=False)

    # Output test image
    date = datetime.datetime.today().strftime("%Y-%m-%d %H%M%S")

    output_dir = args.out
    util.make_dir(output_dir)

    data_n = len(test)
    for j in range(data_n):
        x = test_itr.next().__getitem__(0)[0]
        y = model(xp.asarray([x]))
        if args.gpu >= 0:
            img = util.output2img(chainer.cuda.to_cpu(y.data))
        else:
            img = util.output2img(y.data) 
        Image.fromarray(img[0]).save('{output_dir}/{date}_img{j}.png'.format(**locals()))

if __name__ == '__main__':
    main()
