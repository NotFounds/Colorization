import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

# Network definition
class Colorization(chainer.Chain):

    def __init__(self, map_n, out_n):
        super(Colorization, self).__init__(
            conv1=L.Convolution2D(None, 1*map_n, 4, 2, 1),
            conv2=L.Convolution2D(None, 2*map_n, 4, 2, 1),
            conv3=L.Convolution2D(None, 4*map_n, 4, 2, 1),
            conv4=L.Convolution2D(None, 8*map_n, 4, 2, 1),
            conv5=L.Convolution2D(None, 8*map_n, 4, 2, 1),
            conv6=L.Convolution2D(None, 8*map_n, 4, 2, 1),
            conv7=L.Convolution2D(None, 16*map_n, 4, 2, 1),
            conv8=L.Convolution2D(None, 16*map_n, 4, 2, 1),

            dconv1=L.Deconvolution2D(None, 16*map_n, 4, 2, 1),
            dconv2=L.Deconvolution2D(None, 16*map_n, 4, 2, 1),
            dconv3=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv4=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv5=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv6=L.Deconvolution2D(None, 4*map_n, 4, 2, 1),
            dconv7=L.Deconvolution2D(None, 2*map_n, 4, 2, 1),
            dconv8=L.Deconvolution2D(None, out_n, 4, 2, 1),
        )
        self.map_n = map_n
        self.out_n = out_n

    def __call__(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(F.relu(x1))
        x3 = self.conv3(F.relu(x2))
        x4 = self.conv4(F.relu(x3))
        x5 = F.dropout(self.conv5(F.relu(x4)))
        x6 = F.dropout(self.conv6(F.relu(x5)))
        x7 = F.dropout(self.conv7(F.relu(x6)))
        x8 = F.dropout(self.conv8(F.relu(x7)))

        y = F.concat((F.dropout(self.dconv1(F.relu(x8))), x7))
        y = F.concat((F.dropout(self.dconv2(F.relu(y))), x6))
        y = F.concat((F.dropout(self.dconv3(F.relu(y))), x5))
        y = F.concat((F.dropout(self.dconv4(F.relu(y))), x4))
        y = F.concat((F.dropout(self.dconv5(F.relu(y))), x3))
        y = F.concat((self.dconv6(F.relu(y)), x2))
        y = F.concat((self.dconv7(F.relu(y)), x1))
        y = F.sigmoid(self.dconv8(F.relu(y)))
        return y

class Evalution(chainer.Chain):
    def __init__(self, predictor):
        self.y = None
        self.loss = None
        self.accuracy = None
        super(Evalution, self).__init__(predictor=predictor)

    def __call__(self, x, t=None):
        self.y = self.predictor(x)
        if t is None:
            return self.y
        self.loss = F.mean_squared_error(self.y, t)
        self.accuracy = 1 - F.mean_absolute_error(self.y, t).data
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
