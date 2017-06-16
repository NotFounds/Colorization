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

1. Install `python3.5`
2. Install `chainer`.
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
Colorization ---- train_256_gray ---- (grayscale images)
                  train_256      ---- (color images)
                  test_256_gray  ---- (grayscale images)
                  model.py
                  train.py
                  test.py
                  util.py
```

## Train
You may have to change some following paramaters in `train.py`.
+ dataset directory path
+ epoch num
+ batch size
+ etc..

```
$ python train.py
``` 

## Test
You may have to change some following paramaters in `test.py`.
+ dataset directory path  
+ trained model/optimizer file path
+ etc..

```
$ python test.py
```

# License
MIT License