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
    parser.add_argument('--alpha', type=float, default=0.001, help='Learing rate for Adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='Momentum term of Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='Momentum term of Adam')
    parser.add_argument('--mapsize', type=int, default=2, help='Base size of convolution map')
    parser.add_argument('--snapshot', action='store_const', const=True, default=False, help='Take snapshot of the trainer/model/optimizer')
    parser.add_argument('--no_out_image', action='store_const', const=True, default=False)
    parser.add_argument('--no_print_log', action='store_const', const=True, default=False)
    args = parser.parse_args()

    # Set up a neural network to train
    model = M.Evalution(M.Colorization(args.mapsize, 3))

    # Setup an optimizer
    def make_optimizer(model, alpha=0.001, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    optimizer = make_optimizer(model, args.alpha, args.beta1, args.beta2)

    # Print parameters
    if not args.no_print_log:
        print('Epoch:     {args.epoch}'.format(**locals()))
        print('BatchSize: {args.batchsize}'.format(**locals()))
        print('Alpha:     {args.alpha}'.format(**locals()))
        print('Beta1:     {args.beta1}'.format(**locals()))

    output_dir = args.out
    util.make_dir(output_dir)
    if not args.no_print_log:
        print('Output:    {output_dir}'.format(**locals()))

    # Setup for GPU
    xp = np
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = chainer.cuda.cupy
        if not args.no_print_log:
            print('GPU:       {args.gpu}'.format(**locals()))
    
    # Load the dataset
    train, test = util.make_dataset(args.dataset, xp.float32)

    # Setup iterater
    train_itr = chainer.iterators.SerialIterator(train, args.batchsize)
    test_itr = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Setup trainer
    updater = training.StandardUpdater(train_itr, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_itr, model, device=args.gpu))
    trainer.extend(extensions.dump_graph(root_name='main/loss', out_name='cg.dot'))
    if not args.no_print_log:
        trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
        trainer.extend(extensions.ProgressBar())
    if args.snapshot:
        trainer.extend(extensions.snapshot())
        trainer.extend(extensions.snapshot_object(model, 'model_snapshot_{.updater.epoch}'))
        trainer.extend(extensions.snapshot_object(optimizer, 'optimizer_snapshot_{.updater.epoch}'))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png', marker=None))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png', marker=None))
    
    trainer.run()
    
    # Save model/optimizer
    date = datetime.datetime.today().strftime("%Y-%m-%d %H%M%S")
    serializers.save_npz('{output_dir}/{date}.state'.format(**locals()), optimizer)
    if args.gpu >= 0:
        serializers.save_npz('{output_dir}/{date}_gpu.model'.format(**locals()), model)
        serializers.save_npz('{output_dir}/{date}_cpu.model'.format(**locals()), copy.deepcopy(model).to_cpu())
    else:
        serializers.save_npz('{output_dir}/{date}_cpu.model'.format(**locals()), model)

    # Image output
    if not args.no_out_image:
        data_n = len(test)
        print(data_n)
        output_itr = chainer.iterators.SerialIterator(test, 1, shuffle=False)
        for j in range(data_n):
            x = output_itr.next().__getitem__(0)[0]
            y = model(xp.asarray([x]))
            if args.gpu >= 0:
                img = util.output2img(chainer.cuda.to_cpu(y.data))
            else:
                img = util.output2img(y.data)
            Image.fromarray(img[0]).save('{output_dir}/{date}_img{j}.png'.format(**locals()))

if __name__ == '__main__':
    main()
