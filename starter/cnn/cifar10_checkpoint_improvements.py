from sklearn.preprocessing import LabelBinarizer
from starter.cnn.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os