from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import cv2
import argparse
import os
import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datasets", required=True, help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

# loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-3]
