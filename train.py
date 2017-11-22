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
    def __init__(self, target, targetEpoch):
        self.target = target
        self.targetEpoch

    def __call__(self, optimizer):
        if optimizer.epoch >= self.targetEpoch:
            return
        for name, param in optimizer.target.namedparams():
            for t in self.target:
                if t in name:
                    grad = param.grad
                    with cuda.get_device(grad):
                        grad = 0

def make_optimizer(model, optimizer=chainer.optimizers.Adam()):
    optimizer.setup(model)
    return optimizer

def setup_trainer(model, optimizer, train, test, epoch, out, name, gpu=-1, no_print_log=False, snapshot=False):
    updater = training.StandardUpdater(train, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    trainer.extend(extensions.LogReport(log_name='log_{name}.json'.format(**locals())))
    trainer.extend(extensions.Evaluator(test, model, device=gpu))
    trainer.extend(extensions.dump_graph(root_name='main/loss', out_name='cg_{name}.dot'.format(**locals())))

    if not no_print_log:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())
    if snapshot:
        trainer.extend(extensions.snapshot())
        trainer.extend(extensions.snapshot_object(model, '{name}_snapshot.model'.format(**locals())))
        trainer.extend(extensions.snapshot_object(optimizer, '{name}_snapshot.state'.format(**locals())))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss_{name}.png'.format(**locals()), marker=None))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy_{name}.png'.format(**locals()), marker=None))
    return trainer

def main():
    print('Chainer Version: ' + chainer.__version__)

    parser = argparse.ArgumentParser(description='NN for Grayscale Image colorization')
    parser.add_argument('--batchsize', '-b', type=int, default=50, help='Number of images in each mini-batch')
    parser.add_argument('--epoch_class', type=int, default=300, help='Number of sweeps over the dataset to train(classification model)')
    parser.add_argument('--epoch_color', '-e', type=int, default=400, help='Number of sweeps over the dataset to train(colorization model)')
    parser.add_argument('--dataset', '-d', default='./train', help='Directory of image files. Default is ./train')
    parser.add_argument('--out', '-o', default='./output', help='Directory to output the result')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (native value indicates CPU)')
    parser.add_argument('--snapshot', action='store_const', const=True, default=False, help='Take snapshot of the trainer/model/optimizer')
    parser.add_argument('--no_out_image', action='store_const', const=True, default=False)
    parser.add_argument('--no_print_log', action='store_const', const=True, default=False)
    parser.add_argument('--del_grad', action='store_const', const=True, default=False)
    args = parser.parse_args()

    # Set up a neural network to train
    model_class = L.Classifier(M.Classification(3))

    # Setup an optimizer
    opt_class = make_optimizer(model_class, chainer.optimizers.AdaGrad())

    # Print parameters
    print('# Epoch Class: {args.epoch_class}'.format(**locals()))
    print('# Epoch Color: {args.epoch_color}'.format(**locals()))
    print('# BatchSize:   {args.batchsize}'.format(**locals()))
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
    trainer_class = setup_trainer(model_class, opt_class, train_itr_class, test_itr_class, args.epoch_class, output_dir, 'class', gpu=args.gpu, no_print_log=args.no_print_log, snapshot=args.snapshot)
    trainer_class.run()

    # Set up a neural network to train
    net = opt_class.target.predictor.to_cpu()
    model_color = M.Evalution(M.Colorization(3, net))

    # Setup an optimizer
    opt_color = make_optimizer(model_color, chainer.optimizers.Adam())

    if args.del_grad:
        opt_color.add_hook(DelGrad(['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8'], int(args.epoch_color * 0.8)))

    # Setup for GPU
    if args.gpu >= 0:
        model_color.to_gpu()

    # Setput iterator
    train_itr_color = chainer.iterators.SerialIterator(train_color, args.batchsize)
    test_itr_color  = chainer.iterators.SerialIterator(test_color, args.batchsize, repeat=False, shuffle=False)

    # Setup trainer
    trainer_color = setup_trainer(model_color, opt_color, train_itr_color, test_itr_color, args.epoch_color, output_dir, 'color', gpu=args.gpu, no_print_log=args.no_print_log, snapshot=args.snapshot)
    trainer_color.run()

    # Save model/optimizer
    date = datetime.datetime.today().strftime("%m-%d_%H%M")
    serializers.save_npz('{output_dir}/class.state'.format(**locals()), opt_class)
    serializers.save_npz('{output_dir}/color.state'.format(**locals()), opt_color)
    if args.gpu >= 0:
        serializers.save_npz('{output_dir}/class.model'.format(**locals()), copy.deepcopy(model_class).to_cpu())
        serializers.save_npz('{output_dir}/color.model'.format(**locals()), copy.deepcopy(model_color).to_cpu())
    else:
        serializers.save_npz('{output_dir}/class.model'.format(**locals()), model_class)
        serializers.save_npz('{output_dir}/color.model'.format(**locals()), model_color)
    print('model/optimizer Saved: {output_dir}/*'.format(**locals()))

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
