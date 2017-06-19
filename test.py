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
    date = datetime.datetime.today().strftime("%Y-%m-%d %H%M%S")
    
    model = M.Colorization(2, 3)
    
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam() #0.0002, 0.5
    optimizer.setup(model)
    
    # Load the dataset
    test_data = './test_256_gray/'
    test = util.read_test_data(test_data)
      
    # Load model/optimizer 
    model_data = '2017-06-15 181134.model'
    opt_data   = '2017-06-15 181134.state'
    serializers.load_npz(model_data, model)
    serializers.load_npz(opt_data, optimizer)
   
    # output test image
    output_dir = './output_images/'
    util.make_dir(output_dir)

    N = test._length
    for j in range(N):
        start = time.time()
        x = test.__getitem__(j)[0]
        y = model(np.asarray([x]))
        img = util.output2img(y.data, True)
        Image.fromarray(img[0]).save('{output_dir}{date}_img{j}.png'.format(**locals()))
        elapsed_time = round(time.time() - start, 5)
        print('time: {elapsed_time}[sec]\t{output_dir}{date}_img{j}.png'.format(**locals()))
    
if __name__ == '__main__':
    main()