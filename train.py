import sys
import copy
import time
import random
import argparse
import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

try:
    from chainer import cuda
except ImportError:
    pass

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import datasets
from chainer import training
from chainer import serializers
from chainer.training import extensions
from PIL import Image
import matplotlib.pyplot as plt
import model as M
import util

class DelGrad(object):
    name = 'DelGrad'
    def __init__(self, target):
        self.target = target

    def __call__(self, optimizer):
        for name, param in optimizer.target.namedparams():
            for t in self.target:
                if t in name:
                    grad = param.grad
                    with cuda.get_device(grad):
                        grad = 0

def main():
    print('Chainer Version: ' + chainer.__version__)

    parser = argparse.ArgumentParser(description='NN for Grayscale Image colorization')
    parser.add_argument('--batchsize', '-b', type=int, default=50, help='Number of images in each mini-batch')
    parser.add_argument('--epoch_class', type=int, default=200, help='Number of sweeps over the dataset to train(classification model)')
    parser.add_argument('--epoch_color', '-e', type=int, default=400, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', '-d', default='./train', help='Directory of image files. Default is ./train')
    parser.add_argument('--out', '-o', default='./output', help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (native value indicates CPU)')
    parser.add_argument('--mapsize', type=int, default=8, help='Base size of convolution map')
    parser.add_argument('--snapshot', action='store_const', const=True, default=False, help='Take snapshot of the trainer/model/optimizer')
    parser.add_argument('--no_out_image', action='store_const', const=True, default=False)
    parser.add_argument('--no_print_log', action='store_const', const=True, default=False)
    parser.add_argument('--del_grad', action='store_const', const=True, default=False)
    args = parser.parse_args()

    # Set up a neural network to train
    model_class = L.Classifier(M.Classification(args.mapsize, 3))

    # Setup an optimizer
    def make_optimizer(model, optimizer=chainer.optimizers.Adam()):
        optimizer.setup(model)
        return optimizer
    opt_class = make_optimizer(model_class, chainer.optimizers.AdaGrad())

    # Print parameters
    print('# Epoch Class: {args.epoch_class}'.format(**locals()))
    print('# Epoch Color: {args.epoch_color}'.format(**locals()))
    print('# BatchSize:   {args.batchsize}'.format(**locals()))
    print('# Mapsize:     {args.mapsize}'.format(**locals()))
    print('# Dataset:     {args.dataset}'.format(**locals()))

    # Create Output directory
    output_dir = args.out
    if args.out == './output':
        output_dir = util.next_dir(args.out)
    util.make_dir(output_dir)
    print('# Output:      {output_dir}'.format(**locals()))

    # Setup for GPU
    xp = np
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model_class.to_gpu()
        xp = chainer.cuda.cupy
        print('# GPU:         {args.gpu}'.format(**locals()))

    # Load the dataset
    seed = random.randint(0, 100)
    print('# Seed         {seed}'.format(**locals()))
    print('Data Loading...')
    start = time.time()
    data_class, data_color = util.make_dataset(args.dataset)
    train_class, test_class = datasets.split_dataset_random(data_class, int(len(data_class) * 0.9), seed=seed)
    train_color, test_color = datasets.split_dataset_random(data_color, int(len(data_color) * 0.9), seed=seed)
    diff = time.time() - start
    print('Finish Loading: {diff}s'.format(**locals()))

    # Setup iterater
    train_itr_class = chainer.iterators.SerialIterator(train_class, args.batchsize)
    test_itr_class  = chainer.iterators.SerialIterator(test_class, args.batchsize, repeat=False, shuffle=False)

    # Setup trainer
    updater_class = training.StandardUpdater(train_itr_class, opt_class, device=args.gpu)
    trainer_class = training.Trainer(updater_class, (args.epoch_class, 'epoch'), out=output_dir)
    trainer_class.extend(extensions.LogReport(log_name='log_class.json'))
    trainer_class.extend(extensions.Evaluator(test_itr_class, model_class, device=args.gpu))
    trainer_class.extend(extensions.dump_graph(root_name='main/loss', out_name='cg_class.dot'))

    if not args.no_print_log:
        trainer_class.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer_class.extend(extensions.ProgressBar())
    if args.snapshot:
        trainer_class.extend(extensions.snapshot())
        trainer_class.extend(extensions.snapshot_object(model_class, 'model_class_snapshot.model'))
        trainer_class.extend(extensions.snapshot_object(opt_class, 'optimizer_class_snapshot.state'))
    if extensions.PlotReport.available():
        trainer_class.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss_class.png', marker=None))
        trainer_class.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy_class.png', marker=None))

    trainer_class.run()


    # Set up a neural network to train
    net = opt_class.target.predictor.to_cpu()
    model_color = M.Evalution(M.Colorization(args.mapsize, 3, net))

    # Setup an optimizer
    opt_color = make_optimizer(model_color, chainer.optimizers.Adam())

    if args.del_grad:
        opt_color.add_hook(DelGrad(['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8']))

    # Setup for GPU
    if args.gpu >= 0:
        model_color.to_gpu()

    # Setput iterator
    train_itr_color = chainer.iterators.SerialIterator(train_color, args.batchsize)
    test_itr_color  = chainer.iterators.SerialIterator(test_color, args.batchsize, repeat=False, shuffle=False)

    # Setup trainer
    updater_color = training.StandardUpdater(train_itr_color, opt_color, device=args.gpu)
    trainer_color = training.Trainer(updater_color, (args.epoch_color, 'epoch'), out=output_dir)
    trainer_color.extend(extensions.LogReport(log_name='log_color.json'))
    trainer_color.extend(extensions.Evaluator(test_itr_color, model_color, device=args.gpu))
    trainer_color.extend(extensions.dump_graph(root_name='main/loss', out_name='cg_class.dot'))

    if not args.no_print_log:
        trainer_color.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer_color.extend(extensions.ProgressBar())
    if args.snapshot:
        trainer_color.extend(extensions.snapshot())
        trainer_color.extend(extensions.snapshot_object(model_color, 'model_color_snapshot.model'))
        trainer_color.extend(extensions.snapshot_object(opt_color, 'optimizer_color_snapshot.state'))
    if extensions.PlotReport.available():
        trainer_color.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss_color.png', marker=None))
        trainer_color.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy_color.png', marker=None))

    trainer_color.run()

    # Save model/optimizer
    date = datetime.datetime.today().strftime("%m-%d_%H%M")
    serializers.save_npz('{output_dir}/{date}_class.state'.format(**locals()), opt_class)
    serializers.save_npz('{output_dir}/{date}_color.state'.format(**locals()), opt_color)
    if args.gpu >= 0:
        serializers.save_npz('{output_dir}/{date}_class_gpu.model'.format(**locals()), model_class)
        serializers.save_npz('{output_dir}/{date}_color_gpu.model'.format(**locals()), model_color)
        serializers.save_npz('{output_dir}/{date}_class_cpu.model'.format(**locals()), copy.deepcopy(model_class).to_cpu())
        serializers.save_npz('{output_dir}/{date}_color_cpu.model'.format(**locals()), copy.deepcopy(model_color).to_cpu())
    else:
        serializers.save_npz('{output_dir}/{date}_class_cpu.model'.format(**locals()), model_class)
        serializers.save_npz('{output_dir}/{date}_color_cpu.model'.format(**locals()), model_color)
    print('model/optimizer Saved: {output_dir}/{date}.*'.format(**locals()))

    # Predict test images and Save output images
    if not args.no_out_image:
        chainer.using_config('train', False)
        data_n = len(test_color)
        out_itr = chainer.iterators.SerialIterator(test_color, 1, repeat=False, shuffle=False)
        for i in range(data_n):
            x = out_itr.next().__getitem__(0)[0]
            y = model_color.predictor(xp.asarray([x]))
            if args.gpu >= 0:
                img = util.output2img(chainer.cuda.to_cpu(y.data)[0])
            else:
                img = util.output2img(y.data[0])
            img.save('{output_dir}/{date}_img{i}.png'.format(**locals()))

if __name__ == '__main__':
    main()
