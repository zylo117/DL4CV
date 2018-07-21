from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.regularizers import l2
from keras import backend as K


class DeeperGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim,
                    padding='same', reg=0.0005, name=None):
        # initialize the CONV, BN, and RELU layer names
        (convName, bnName, actName) = (None, None, None)

        # if a layer name was supplied, prepend it
        if name is not None:
            convName = name + '_conv'
            bnName = name + '_bn'
            actName = name + '_act'

        # define a CONV => BN => RELU pattern
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding,
                   kernel_regularizer=l2(reg), name=convName)(x)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)
        x = Activation('relu', name=actName)(x)

        # return the block
        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3Reduce, num3x3,
                         num5x5Reduce, num5x5, num1x1Proj, chanDim,
                         stage, reg=0.0005):
        # define the first branch of the Inception module which
        # consists of 1x1 convolutions
        first = DeeperGoogLeNet.conv_module(x, num1x1, 1, 1,
                                            (1, 1), chanDim, reg=reg,
                                            name=stage + '_first')

        # define the second branch of the Inception module which
        # consists of 1x1 and 3x3 convolutions
        # The number of 1x1 convolutions is always smaller than the number
        # of 3x3 convolutions, thereby serving as a form of dimensionality reduction.
        second = DeeperGoogLeNet.conv_module(x, num3x3Reduce, 1, 1,
                                             (1, 1), chanDim, reg=reg,
                                             name=stage + '_second1')
        second = DeeperGoogLeNet.conv_module(second, num3x3, 3, 3,
                                             (1, 1), chanDim, reg=reg,
                                             name=stage + '_second2')

        # define the third branch of the Inception module which
        # are our 1x1 and 5x5 convolutions
        third = DeeperGoogLeNet.conv_module(x, num5x5Reduce, 1, 1,
                                             (1, 1), chanDim, reg=reg,
                                             name=stage + '_third1')
        third = DeeperGoogLeNet.conv_module(second, num5x5, 5, 5,
                                             (1, 1), chanDim, reg=reg,
                                             name=stage + '_third2')
