# alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
import matplotlib

from tools.multi_gpu import ParallelStandardModel

matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from starter.cnn.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse

G = 1

def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
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

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# define the set of callbacks to be passed to the model during training
callbacks = [LearningRateScheduler(step_decay)]

# initialize the optimizer and model
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
# check to see if we are compiling using just a single GPU
if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)

# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    with tf.device("/cpu:0"):
        # initialize the model
        model = MiniVGGNet.build(width=32, height=32, depth=3,classes=10)

        # make the model parallel
        model = multi_gpu_model(model, gpus=G)

model.summary()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64 * G, epochs=40, callbacks=callbacks, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64 * G)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(40), H.history["loss"], label="train_loss")
plt.plot(np.arange(40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(40), H.history["acc"], label="train_acc")
plt.plot(np.arange(40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig(args["output"])