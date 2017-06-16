import sys
import locale
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
    start = time.time()
    date = datetime.datetime.today().strftime("%Y-%m-%d %H%M%S")

    # Set up a neural network to train
    model = M.Colorization(2, 3)
    
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam() #0.0002, 0.5
    optimizer.setup(model)
    
    # Load the dataset
    train_in  = './train_256_gray/'
    train_out = './train_256/'
    test_in   = './test_256_gray/'
    
    train = util.read_train_data(train_in, train_out)
    test  = util.read_test_data(test_in) 

    N = train._length
    n_epoch    = 1000
    batch_size = 50
    print('DataSets:  {N}'.format(**locals()))
    print('Epoch:     {n_epoch}'.format(**locals()))
    print('BatchSize: {batch_size}'.format(**locals()))

    losses = []
    for epoch in range(n_epoch):
        perm = np.random.permutation(N)
        sum_loss = np.float32(0)
        
        for i in range(batch_size):
            idx = perm[i]
            x = train.__getitem__(idx)[0]
            y = model(np.asarray([x]))
            t = train.__getitem__(idx)[1]
            loss = F.mean_squared_error(y[0], t)
            optimizer.zero_grads()
            loss.backward()
            optimizer.update()
            sum_loss += loss.data

        sum_loss /= batch_size
        losses.append([epoch, sum_loss])
        #sys.stderr.write("epoch: {epoch}\t\tloss: {sum_loss}\n".format(**locals()))
        print("epoch: {epoch}\t\tloss: {sum_loss}".format(**locals()))
    
    test_N = test._length
    for j in range(test_N):
        x = test.__getitem__(j)[0]
        y = model(np.asarray([x]))
        img = util.output2img(y.data)
        Image.fromarray(img[0]).save('{date}_img{j}.png'.format(**locals()))

    # save model/optimizer
    serializers.save_npz('{date}.model'.format(**locals()), model)
    serializers.save_npz('{date}.state'.format(**locals()), optimizer)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    e = [d[0] for d in losses]
    l = [d[1] for d in losses]
    plt.plot(e, l, color="red")
    plt.savefig('{date}_graph.jpg'.format(**locals()))
    
if __name__ == '__main__':
    main()