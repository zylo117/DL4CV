# set the matplotlib backend so figures can be saved in the background
# alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
import matplotlib

matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from cnn.nn.conv.minivggnet import MiniVGGNet
from cnn.callbacks.trainingmonitor import TrainMonitor
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.datasets import cifar10
from keras.backend import set_session
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# config GPU
G = 2
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# show the infomation on the process ID
print("[INFO] process ID: %s" % os.getpid())

# load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX / 255
testX = testX / 255

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# initialize the SGD optimizer, but without any learning rate decay
print("[INFO] compiling lpr_model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
# check to see if we are compiling using just a single GPU
if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)

# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the lpr_model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)

    # make the lpr_model parallel
    model = multi_gpu_model(model, gpus=G)

model.summary()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "%s.png" % os.getpid()])
jsonPath = os.path.sep.join([args["output"], "%s.json" % os.getpid()])
callbacks = [TrainMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64 * G, epochs=100, callbacks=callbacks, verbose=1)
