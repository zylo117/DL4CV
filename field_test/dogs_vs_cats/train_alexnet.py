# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use('Agg')

from cnn.nn.conv.alexnet import AlexNet
from field_test.dogs_vs_cats.config import dogs_vs_cats_config as config
from cnn.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from cnn.preprocessing.simplepreprocessor import SimplePreprocessor
from cnn.preprocessing.patchpreprocessor import PatchPreprocessor
from cnn.preprocessing.meanpreprocessor import MeanPreprocessor
from cnn.callbacks.trainingmonitor import TrainMonitor
from cnn.io_.hdf5datasetgenrator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend import set_session
from keras.utils import multi_gpu_model
import json
import os

G = 2
if G > 1:
    print("[INFO] setting up for multi-gpu")
    gm_config = tf.ConfigProto()
    gm_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=gm_config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug,
                                preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128,
                              preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
print('[INFO] compiling lpr_model...')
opt = Adam(lr=1e-3)
single_gpu_model = AlexNet.build(width=227, height=227, depth=3,
                                 classes=2, reg=0.0002)

if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = single_gpu_model
    # otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the lpr_model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    # make the lpr_model parallel
    model = multi_gpu_model(single_gpu_model, gpus=G)

model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, '{}.png'.format(os.getpid())])
callbacks = [TrainMonitor(path)]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 128,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 128,
    epochs=75,
    max_queue_size=128 * 2,
    callbacks=callbacks, verbose=1
)

# save the lpr_model to file
print('[INFO] serializing lpr_model...')
single_gpu_model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()
