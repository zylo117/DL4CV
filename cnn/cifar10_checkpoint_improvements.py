import warnings

from keras.backend import set_session
from keras.utils import multi_gpu_model
from sklearn.preprocessing import LabelBinarizer
from cnn.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import argparse
import os

# config GPU
from tools.multi_gpu import ParallelModelCheckpoint

G = 1
if G > 1:
    print("[INFO] setting up for multi-gpu")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to weights directory")
ap.add_argument("-b", "--best_only", type=bool, default=True, help="If True, model will only write a single file with best result")
args = vars(ap.parse_args())

# load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX / 255
testX = testX / 255

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)

# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    single_gpu_model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)

    # make the model parallel
    model = multi_gpu_model(single_gpu_model, gpus=G)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the callback to save only the *best* model to disk
# based on the validation loss
if not args["best_only"]:
    fname = os.path.sep.join([args["weights"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
else:
    fname = args["weights"]

if G <= 1:
    print("[INFO] outputing model checkpoints...")
    checkpoint = ModelCheckpoint(filepath=fname, monitor="val_loss", mode="min",
                                 save_best_only=True, verbose=1)
else:
    print("[INFO] outputing parallel model checkpoints...")
    checkpoint = ParallelModelCheckpoint(single_gpu_model, filepath=fname, monitor="val_loss", mode="min",
                                         save_best_only=True, save_weights_only=False, verbose=1)
callbacks = [checkpoint]

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
