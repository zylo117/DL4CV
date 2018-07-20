import datetime
import cv2
import json
from keras.backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

import field_test.smart_test.find_tripod.config.tripod_config as config
from cnn.io_.hdf5datasetgenrator import HDF5DatasetGenerator
from cnn.nn.conv.alexnet import AlexNet
from cnn.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from cnn.preprocessing.meanpreprocessor import MeanPreprocessor
from cnn.preprocessing.patchpreprocessor import PatchProcessor
from cnn.preprocessing.preprocess import SimplePreprocessor
from tools.multi_gpu import ParallelModelCheckpoint

EPOCHS = 100
LEARNINGRATE = 1e-3
BATCHSIZE = 32

G = 2
if G > 1:
    print("[INFO] setting up for multi-gpu")
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=tf_config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to weights directory")
ap.add_argument("-b", "--best_only", type=bool, default=True,
                help="If True, model will only write a single file with best result")
ap.add_argument("-v", "--view", help="save image of training status")
args = vars(ap.parse_args())

# load images
print("[INFO] loading images")
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest')

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227, inter=cv2.INTER_LANCZOS4)
pp = PatchProcessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, BATCHSIZE * 2, aug=aug,
                                preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, BATCHSIZE * 2,
                              preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
print('[INFO] compiling model...')
opt = Adam(lr=LEARNINGRATE)
single_gpu_model = AlexNet.build(width=227, height=227, depth=3,
                                 classes=2, reg=0.0002)

if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = single_gpu_model
    # otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    # make the model parallel
    model = multi_gpu_model(single_gpu_model, gpus=G)

model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# construct the callback to save only the *best* model to disk
# based on the validation loss
if not args["best_only"]:
    fname = os.path.sep.join([args["weights"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
else:
    fname = os.path.sep.join([args["weights"], "{}.hdf5".format(datetime.date.today())])

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
H = model.fit_generator(
    generator=trainGen.generator(),
    steps_per_epoch=trainGen.numImages // (BATCHSIZE * 2),
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // (BATCHSIZE * 2),
    epochs=EPOCHS,
    max_queue_size=BATCHSIZE * G,
    callbacks=callbacks, verbose=1
)

# save the model to file
print('[INFO] serializing model...')
single_gpu_model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["view"])
plt.show()
