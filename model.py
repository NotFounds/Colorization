import chainer
import chainer.functions as F
import chainer.links as L

# Network definition
class Colorization(chainer.Chain):

    def __init__(self, map_n, out_n):
        super(Colorization, self).__init__(
            conv1=L.Convolution2D(None, map_n, 4, 2, 1),
            conv2=L.Convolution2D(None, 2*map_n, 4, 2, 1),
            conv3=L.Convolution2D(None, 4*map_n, 4, 2, 1),
            conv4=L.Convolution2D(None, 8*map_n, 4, 2, 1),
            conv5=L.Convolution2D(None, 8*map_n, 4, 2, 1),
            conv6=L.Convolution2D(None, 8*map_n, 4, 2, 1),
            conv7=L.Convolution2D(None, 8*map_n, 4, 2, 1),
            conv8=L.Convolution2D(None, 8*map_n, 4, 2, 1),

            bn2=L.BatchNormalization(2*map_n),
            bn3=L.BatchNormalization(4*map_n),
            bn4=L.BatchNormalization(8*map_n),
            bn5=L.BatchNormalization(8*map_n),
            bn6=L.BatchNormalization(8*map_n),
            bn7=L.BatchNormalization(8*map_n),
            bn8=L.BatchNormalization(8*map_n),

            dconv1=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv2=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv3=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv4=L.Deconvolution2D(None, 8*map_n, 4, 2, 1),
            dconv5=L.Deconvolution2D(None, 4*map_n, 4, 2, 1),
            dconv6=L.Deconvolution2D(None, 2*map_n, 4, 2, 1),
            dconv7=L.Deconvolution2D(None, map_n, 4, 2, 1),
            dconv8=L.Deconvolution2D(None, out_n, 4, 2, 1),

            dbn1=L.BatchNormalization(8*map_n),
            dbn2=L.BatchNormalization(8*map_n),
            dbn3=L.BatchNormalization(8*map_n),
            dbn4=L.BatchNormalization(8*map_n),
            dbn5=L.BatchNormalization(4*map_n),
            dbn6=L.BatchNormalization(2*map_n),
            dbn7=L.BatchNormalization(map_n),
        )
        self.map_n = map_n
        self.out_n = out_n

    def __call__(self, x):
        x1 = self.conv1(x)
        x2 = self.bn2(self.conv2(F.relu(x1)))
        x3 = self.bn3(self.conv3(F.relu(x2)))
        x4 = self.bn4(self.conv4(F.relu(x3)))
        x5 = self.bn5(self.conv5(F.relu(x4)))
        x6 = self.bn6(self.conv6(F.relu(x5)))
        x7 = self.bn7(self.conv7(F.relu(x6)))
        x8 = self.bn8(self.conv8(F.relu(x7)))

        y1 = F.concat((F.dropout(self.dbn1(self.dconv1(F.relu(x8)))), x7))
        y2 = F.concat((F.dropout(self.dbn2(self.dconv2(F.relu(y1)))), x6))
        y3 = F.concat((F.dropout(self.dbn3(self.dconv3(F.relu(y2)))), x5))
        y4 = F.concat((self.dbn4(self.dconv4(F.relu(y3))), x4))
        y5 = F.concat((self.dbn5(self.dconv5(F.relu(y4))), x3))
        y6 = F.concat((self.dbn6(self.dconv6(F.relu(y5))), x2))
        y7 = F.concat((self.dbn7(self.dconv7(F.relu(y6))), x1))
        y8 = F.tanh(self.dconv8(F.relu(y7)))
        return y8
