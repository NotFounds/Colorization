# Colorization
A simple colorization neural network.

# Feature
+ Using `Chainer`
+ Using CNN(Convolutional Neural Network)
+ Using 8 convolution layers, and 8 deconvolution layers
+ No pooling layers
+ No fully connected layers
+ Able to use `GPU`

# Instrations
3 steps to install easily.

1. Install [Python3.5](https://www.python.org/).
2. Install [Chainer](https://chainer.org/).
3. Clone this [repo](https://github.com/NotFounds/Colorization).

```
$ git clone https://github.com/NotFounds/Colorization.git
$ cd Colorization
```

# Usage
## File Hierarchy
Prepare some grayscale images and corresponging color images.  
And resize imeges to 256 * 256.
```
Colorization ---- train ---- (color images) --> train: 90%
              |                             +-> test:  10% 
              +-- test ----- (gray images)
              +-- model.py
              +-- train.py
              +-- test.py
              +-- util.py
```

## Train
You may have to change some following paramaters in `train.py`.
```
$ python train.py [options]
```
| option            | type  | description                                            |
| ----------------- | ----- | ------------------------------------------------------ |
| --batchsize, -b   | int   | batch size. default is 50.                             |
| --epoch, -e       | int   | epoch num. default is 400.                             |
| --dataset, -d     | path  | the directory path of train data. default is `./train`.|
| --out, -o         | path  | the directory path of output. default is `./output`.   |
| --gpu, -g         | int   | gpu id. default is -1.(no gpu)                         |
| --alpha           | float | learning rate fot Adam. default is 0.001.              |
| --beta1           | float | momentum term of Adam. default is 0.9.                 |
| --beta2           | float | momentum term of Adam. default is 0.999.               |
| --mapsize         | int   | base size of convolution map.                          |
| --snapshot        | None  | take snapshot of the trainer/model/optimizer.          |
| --no_out_image    | None  | don't output images.                                   |
| --no_print_log    | None  | don't print log.                                       |

## Test
You may have to change some following paramaters in `test.py`.
```
$ python test.py [options]
```
| option            | type  | description                                                      |
| ----------------- | ----- | ---------------------------------------------------------------- |
| --dataset, -d     | path  | the directory path of input data. default is `./test`.           |
| --out, -o         | path  | the directory path of output. default is `./output`.             |
| --model, -m       | path  | the file path of learned NN model. default is `./example.model`. |
| --mapsize         | None  | base size of convolution map.                                    |

# License
MIT License