from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from cnn.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from cnn.preprocessing.preprocess import SimplePreprocessor
from cnn.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]