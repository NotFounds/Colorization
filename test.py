import argparse
import datetime
import time
import numpy as np
import chainer
from chainer import serializers
from PIL import Image
import model as M
import util

def main():
    print(chainer.__version__)

    parser = argparse.ArgumentParser(description='Colorization')
    parser.add_argument('--data', '-d', default='./test/gray')
    parser.add_argument('--model', '-m', default='./example.model')
    parser.add_argument('--out', '-o', default='./output')
    args = parser.parse_args()

    date = datetime.datetime.today().strftime("%Y-%m-%d %H%M%S")

    # Set up a neural network
    model = M.Colorization(2, 3)

    # Load the dataset
    test_data = args.data + '/'
    test = util.read_test_data(test_data)

    # Load model
    model_data = args.model
    serializers.load_npz(model_data, model)

    # output test image
    output_dir = args.out
    util.make_dir(output_dir)

    data_n = len(test)
    for j in range(data_n):
        start = time.time()
        x = test.__getitem__(j)[0]
        y = model(np.asarray([x]))
        img = util.output2img(y.data)
        Image.fromarray(img[0]).save('{output_dir}/{date}_img{j}.png'.format(**locals()))
        elapsed_time = round(time.time() - start, 5)
        print('time: {elapsed_time}[sec]\t{output_dir}/{date}_img{j}.png'.format(**locals()))

if __name__ == '__main__':
    main()
