import datetime
import time
import numpy as np
import chainer
import chainer.functions as F
from chainer import serializers
from PIL import Image
import matplotlib.pyplot as plt
import model as M
import util

def main():
    print(chainer.__version__)

    start = time.time()
    date = datetime.datetime.today().strftime("%Y-%m-%d %H%M%S")

    # Set up a neural network to train
    model = M.Colorization(2, 3)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the dataset
    train_in = './train_256_gray/'
    train_out = './train_256/'
    test_in = './test_256_gray/'

    train = util.read_train_data(train_in, train_out)
    test = util.read_test_data(test_in)

    data_n = len(train)
    epoch_n = 1000
    batch_n = 50
    print('DataSets:  {data_n}'.format(**locals()))
    print('Epoch:     {epoch_n}'.format(**locals()))
    print('BatchSize: {batch_n}'.format(**locals()))

    losses = []
    for epoch in range(epoch_n):
        perm = np.random.permutation(data_n)
        sum_loss = np.float32(0)

        for i in range(batch_n):
            idx = perm[i]
            x = train.__getitem__(idx)[0]
            y = model(np.asarray([x]))
            t = train.__getitem__(idx)[1]

            optimizer.use_cleargrads()
            loss = F.mean_squared_error(y[0], t)
            loss.backward()
            optimizer.update(F.mean_squared_error, y[0], t)
            sum_loss += loss.data

        sum_loss /= batch_n
        losses.append([epoch, sum_loss])
        #sys.stderr.write("epoch: {epoch}\t\tloss: {sum_loss}\n".format(**locals()))
        print("epoch: {epoch}\t\tloss: {sum_loss}".format(**locals()))

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # save loss graph
    x_val = [d[0] for d in losses]
    _loss = [d[1] for d in losses]
    plt.plot(x_val, _loss, color="red")
    plt.savefig('{date}_graph.jpg'.format(**locals()))

    # save model/optimizer
    serializers.save_npz('{date}.model'.format(**locals()), model)
    serializers.save_npz('{date}.state'.format(**locals()), optimizer)

    # output test image
    output_dir = './output_images/'
    util.make_dir(output_dir)

    test_n = len(test)
    for j in range(test_n):
        x = test.__getitem__(j)[0]
        y = model(np.asarray([x]))
        img = util.output2img(y.data)
        Image.fromarray(img[0]).save('{output_dir}{date}_img{j}.png'.format(**locals()))

if __name__ == '__main__':
    main()
