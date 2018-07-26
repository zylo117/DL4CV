import matplotlib
from keras.backend import set_session
from keras.utils import multi_gpu_model

matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from cnn.nn.conv.minigooglenet import MiniGoogLeNet
from cnn.callbacks.trainingmonitor import TrainMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import argparse
import os

# define the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 70
INIT_LR = 1e-1

G = 4
if G > 1:
    print("[INFO] setting up for multi-gpu")
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # set_session(tf.Session(config=config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
                help='path to output model')
ap.add_argument('-o', '--output', required=True,
                help='path to output directory (logs, plots, etc.)')
args = vars(ap.parse_args())


aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest')

# load the training and testing data, converting the images from
# integers to floats
print('[INFO] loading CIFAR-10 data...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype(np.float)
testX = testX.astype(np.float)

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

# construct the set of callbacks
figPath = os.path.sep.join([args['output'], '{}.png'.format(
    os.getpid())])
jsonPath = os.path.sep.join([args['output'], '{}.json'.format(
    os.getpid())])
callback = [TrainMonitor(figPath, jsonPath=jsonPath),
            LearningRateScheduler(poly_decay)]

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=INIT_LR, momentum=0.9, nesterov=True)

single_gpu_model = MiniGoogLeNet.build(width=32, height=32,
                                       depth=3, classes=10)

if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = single_gpu_model
# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))
    # make the model parallel
    model = multi_gpu_model(single_gpu_model, gpus=G)

model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# train the network
print('[INFO] training network...')
model.fit_generator(aug.flow(trainX, trainY, batch_size=128 * G),
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // (128 * G),
                    epochs=NUM_EPOCHS, callbacks=callback, verbose=1)

# save the network to disk
print('[INFO] serializing network...')
if G <= 1:
    model.save(args["model"])
else:
    single_gpu_model.save(args["model"])