from keras.backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from cnn.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from cnn.datasets.dataset_loader import SimpleDatasetLoader
from cnn.nn.conv.fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import os

from tools.multi_gpu import ParallelModelCheckpoint
from tools.paths import list_images

EPOCHS = 150

G = 2
if G > 1:
    print("[INFO] setting up for multi-gpu")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--lpr_model", required=True, help="path to output lpr_model")
ap.add_argument("-w", "--weights", required=True, help="path to weights directory")
ap.add_argument("-b", "--best_only", type=bool, default=True,
                help="If True, lpr_model will only write a single file with best result")
args = vars(ap.parse_args())

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode="nearest")

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list_images(args["dataset"])

# initialize the image preprocessors
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to
# the range [0, 1]
sdl = SimpleDatasetLoader(preprocessor=[iap])
data, labels = sdl.load(imagePaths, verbose=500)
data = data / 255
classNames = [str(x) for x in np.unique(labels)]

# convert the labels from integers to vectors
labels = LabelBinarizer().fit_transform(labels)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels,
                                                test_size=0.25,
                                                stratify=labels,
                                                random_state=42)

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 512)

# place the head FC lpr_model on top of the base lpr_model -- this will
# become the actual lpr_model we will train

if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = Model(inputs=baseModel.input, outputs=headModel)
# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the lpr_model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    single_gpu_model = Model(inputs=baseModel.input, outputs=headModel)
    # make the lpr_model parallel
    model = multi_gpu_model(single_gpu_model, gpus=G)

# loop over all layers in the base lpr_model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our lpr_model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling lpr_model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# construct the callback to save only the *best* lpr_model to disk
# based on the validation loss
if not args["best_only"]:
    fname = os.path.sep.join([args["weights"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
else:
    fname = args["weights"]

if G <= 1:
    print("[INFO] outputing lpr_model checkpoints...")
    checkpoint = ModelCheckpoint(filepath=fname, monitor="val_loss", mode="min",
                                 save_best_only=True, verbose=1)
else:
    print("[INFO] outputing parallel lpr_model checkpoints...")
    checkpoint = ParallelModelCheckpoint(single_gpu_model, filepath=fname, monitor="val_loss", mode="min",
                                         save_best_only=True, save_weights_only=False, verbose=1)
callbacks = [checkpoint]

# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual "learned" values
# versus pure random
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY), epochs=int(EPOCHS / 4),
                    callbacks=callbacks,
                    class_weight=classWeight,
                    steps_per_epoch=len(trainX) // 32, verbose=1)

# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
    layer.trainable = True

# for the changes to the lpr_model to take affect we need to recompile
# the lpr_model, this time using SGD with a *very* small learning rate
print("[INFO] re-compiling lpr_model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the lpr_model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
print("[INFO] fine-tuning lpr_model...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
                    callbacks=callbacks,
                    class_weight=classWeight,
                    epochs=EPOCHS, steps_per_epoch=len(trainX) // 32, verbose=1)

# evaluate the network on the fine-tuned lpr_model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

# save the lpr_model to disk
print("[INFO] serializing lpr_model...")
if G <= 1:
    model.save(args["lpr_model"])
else:
    single_gpu_model.save(args["lpr_model"])

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
plt.show()
