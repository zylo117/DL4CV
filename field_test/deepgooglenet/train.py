# set the matplotlib backend so figures can be saved in the background
import os

import matplotlib
matplotlib.use('Agg')

from keras.utils import multi_gpu_model
import field_test.deepgooglenet.config.tiny_imagenet_config as config
from cnn.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from cnn.preprocessing.simplepreprocessor import SimplePreprocessor
from cnn.preprocessing.meanpreprocessor import MeanPreprocessor
from cnn.callbacks.parallelmodelcheckpoint import ParallelModelCheckpoint
from cnn.callbacks.epochcheckpoint import EpochCheckpoint
from cnn.callbacks.trainingmonitor import TrainMonitor
from cnn.io_.hdf5datasetgenrator import HDF5DatasetGenerator
from cnn.nn.conv.deepergooglenet import DeeperGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import argparse
import json

EPOCHS = 70

G = 2
if G > 1:
    print("[INFO] setting up for multi-gpu")
    gm_config = tf.ConfigProto()
    gm_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    K.set_session(tf.Session(config=gm_config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True,
                help='path to output checkpoint directory')
ap.add_argument('-m', '--model', type=str,
                help='path to *specific* model checkpoint to load')
ap.add_argument('-s', '--start-epoch', type=int, default=0,
                help='epoch to restart training at')
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest')

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
                                preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
                              preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

opt = Adam(1e-3)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args['model'] is None:
    print('[INFO] compiling model...')
    single_gpu_model = DeeperGoogLeNet.build(64, 64, 3,
                                  classes=config.NUM_CLASSES,
                                  reg=0.0002)

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

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

# otherwise, load the checkpoint from disk
else:
    print('[INFO] loading {}...'.format(args['model']))
    single_gpu_model = load_model(args['model'], compile=False)

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

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    # update the learning rate
    print('[INFO] old learning rate: {}'.format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print('[INFO] new learning rate:{}'.format(
        K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
    EpochCheckpoint(args['checkpoints'], every=5,
                    startAt=args['start_epoch'],
                    how_many_gpus=G, single_gpu_model=single_gpu_model),
    TrainMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH,
                 startAt=args['start_epoch'])
]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 64,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 64,
    epochs=EPOCHS,
    max_queue_size=64 * 2,
    callbacks=callbacks,
    verbose=1
)

# close the databases
trainGen.close()
valGen.close()