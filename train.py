import sys
import copy
import argparse
import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import chainer
import chainer.functions as F
from chainer import serializers
from chainer import training
from chainer.training import extensions
from PIL import Image
import matplotlib.pyplot as plt
import model as M
import util

def main():
    print('Chainer Version: ' + chainer.__version__)

    parser = argparse.ArgumentParser(description='NN for Grayscale Image colorization')
    parser.add_argument('--batchsize', '-b', type=int, default=50, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', '-d', default='./train', help='Directory of image files. Default is ./train')
    parser.add_argument('--out', '-o', default='./output', help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (native value indicates CPU)')
    parser.add_argument('--alpha', type=float, default=0.0002, help='Learing rate for Adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term of Adam')
    args = parser.parse_args()

    # Set up a neural network to train
    model = M.Evalution(M.Colorization(2, 3))

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        return optimizer
    optimizer = make_optimizer(model, args.alpha, args.beta1)

    # Print parameters
    print('Epoch:     {args.epoch}'.format(**locals()))
    print('BatchSize: {args.batchsize}'.format(**locals()))
    print('Alpha:     {args.alpha}'.format(**locals()))
    print('Beta1:     {args.beta1}'.format(**locals()))

    output_dir = args.out
    util.make_dir(output_dir)
    print('Output:    {output_dir}'.format(**locals()))

    # Setup for GPU
    xp = np
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = chainer.cuda.cupy
        print('GPU:       {args.gpu}'.format(**locals()))
    
    # Load the dataset
    train, test = util.make_dataset(args.dataset, xp.float32) 

    # Setup iterater
    train_itr = chainer.iterators.SerialIterator(train, args.batchsize)
    test_itr = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Setup trainer
    updater = training.StandardUpdater(train_itr, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)

    trainer.extend(extensions.Evaluator(test_itr, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.dump_graph(root_name='main/loss', out_name='cg.dot'))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png', marker=None))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png', marker=None))
    
    trainer.run()
    
    # save model/optimizer
    date = datetime.datetime.today().strftime("%Y-%m-%d %H%M%S")
    serializers.save_npz('{output_dir}/{date}.state'.format(**locals()), optimizer)
    if args.gpu >= 0:
        serializers.save_npz('{output_dir}/{date}_gpu.model'.format(**locals()), model)
        serializers.save_npz('{output_dir}/{date}_cpu.model'.format(**locals()), copy.deepcopy(model).to_cpu())
    else:
        serializers.save_npz('{output_dir}/{date}_cpu.model'.format(**locals()), model)

if __name__ == '__main__':
    main()
