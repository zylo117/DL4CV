from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from starter.cnn.preprocessing.preprocess import ImageToArrayPreprocessor
from starter.cnn.preprocessing.preprocess import SimplePreprocessor
from starter.cnn.datasets.dataset_loader import SimpleDatasetLoader
from starter.cnn.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse