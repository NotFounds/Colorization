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

            bn2=L.BatchNormalization(2*map_n),
            bn3=L.BatchNormalization(4*map_n),
            bn4=L.BatchNormalization(8*map_n),
            bn5=L.BatchNormalization(8*map_n),
            bn6=L.BatchNormalization(8*map_n),
            bn7=L.BatchNormalization(16*map_n),
            bn8=L.BatchNormalization(16*map_n),

            dconv1=L.Deconvolution2D(None, 16*map_n, 4, 2, 1),
            dconv2=L.Deconvolution2D(None, 16*map_n, 4, 2, 1),
            dconv3=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv4=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv5=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv6=L.Deconvolution2D(None, 4*map_n, 4, 2, 1),
            dconv7=L.Deconvolution2D(None, 2*map_n, 4, 2, 1),
            dconv8=L.Deconvolution2D(None, out_n, 4, 2, 1),

            dbn1=L.BatchNormalization(16*map_n),
            dbn2=L.BatchNormalization(16*map_n),
            dbn3=L.BatchNormalization(8*map_n),
            dbn4=L.BatchNormalization(8*map_n),
            dbn5=L.BatchNormalization(8*map_n),
            dbn6=L.BatchNormalization(4*map_n),
            dbn7=L.BatchNormalization(2*map_n),
        )
        self.map_n = map_n
        self.out_n = out_n

    def __call__(self, x):
        x1 = self.conv1(x)
        x2 = self.bn2(self.conv2(F.relu(x1)))
        x3 = self.bn3(self.conv3(F.relu(x2)))
        x4 = self.bn4(self.conv4(F.relu(x3)))
        x5 = F.dropout(self.bn5(self.conv5(F.relu(x4))), ratio=0.9)
        x6 = F.dropout(self.bn6(self.conv6(F.relu(x5))), ratio=0.9)
        x7 = F.dropout(self.bn7(self.conv7(F.relu(x6))), ratio=0.9)
        x8 = F.dropout(self.bn8(self.conv8(F.relu(x7))), ratio=0.9)

        y1 = F.concat((F.dropout(self.dbn1(self.dconv1(F.relu(x8))), ratio=0.7), x7))
        y2 = F.concat((F.dropout(self.dbn2(self.dconv2(F.relu(y1))), ratio=0.7), x6))
        y3 = F.concat((F.dropout(self.dbn3(self.dconv3(F.relu(y2))), ratio=0.7), x5))
        y4 = F.concat((F.dropout(self.dbn4(self.dconv4(F.relu(y3))), ratio=0.8), x4))
        y5 = F.concat((F.dropout(self.dbn5(self.dconv5(F.relu(y4))), ratio=0.9), x3))
        y6 = F.concat((self.dbn6(self.dconv6(F.relu(y5))), x2))
        y7 = F.concat((self.dbn7(self.dconv7(F.relu(y6))), x1))
        y8 = F.sigmoid(self.dconv8(F.relu(y7)))
        return y8

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
        self.accuracy = 1 - F.mean_squared_error(self.y, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
