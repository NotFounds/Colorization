# Colorization
A simple colorization neural network.

# Feature
+ Using `Chainer`
+ Using CNN(Convolutional Neural Network)
+ Using 8 convolution layers, and 8 deconvolution layers
+ No pooling layers
+ No fully connected layers

# Example
| Grayscale Image | Output Image | Original Image |
| -------------- | --------------- | ------------ |
|![example1_gray](./examples/example1_gray.jpg)|![example1_out](./examples/example1_out.png)|![example1_origin](./examples/example1_origin.jpg)|
|![example2_gray](./examples/example2_gray.jpg)|![example2_out](./examples/example2_out.png)|![example2_origin](./examples/example2_origin.jpg)|
|![example3_gray](./examples/example3_gray.jpg)|![example3_out](./examples/example3_out.png)|![example3_origin](./examples/example3_origin.jpg)|

# Instrations
3 steps to install easily.

1. Install [python3.5](https://www.python.org/).
2. Install [chainer](https://chainer.org/).
3. Clone this repo.

```
$ git clone https://github.com/NotFounds/Colorization.git
$ cd Colorization
```

# Usage
## File Hierarchy
Prepare some grayscale images and corresponging color images.  
And resize imeges to 256 * 256.
```
Colorization ---- train ---- gray ---- (grayscale images)
              |           +- origin -- (color images)
              +-- test ----- gray ---- (grayscale images)
              |           +- origin -- (color images)
              +-- model.py
              +-- train.py
              +-- test.py
              +-- util.py
```

## Train
You may have to change some following paramaters in `train.py`.

| option            | type | description                                            |
| ----------------- | ---- | ------------------------------------------------------ |
| --batchsize, -b   | int  | batch size. default is 50.                             |
| --epoch, -e       | int  | epoch num. default is 1000.                            |
| --train           | path | the directory path of train data. default is `./train`.|
| --test            | path | the directory path of test data. default is `./test`.  |
| --out, -o         | path | the directory path of output. default is `./output`.   |
| --debug, -d       | -    | debug option. default is false.                        |

```
$ python train.py [options]
``` 

## Test
You may have to change some following paramaters in `test.py`.

| option            | type | description                                                      |
| ----------------- | ---- | ---------------------------------------------------------------- |
| --data, -d        | path | the directory path of input data. default is `./test/gray`.      |
| --out, -o         | path | the directory path of output. default is `./output`.             |
| --model, -m       | path | the file path of learned NN model. default is `./example.model`. |

```
$ python test.py [options]
```

# License
MIT License