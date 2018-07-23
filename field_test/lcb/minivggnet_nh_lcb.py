from keras.backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model, np_utils
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from cnn.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from cnn.datasets.dataset_loader import SimpleDatasetLoader
from cnn.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from cnn.nn.conv.minivggnet import MiniVGGNet
from cnn.nn.conv.microvggnet import MicroVGGNet
from cnn.nn.conv.vggnet import VGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import os

from tools.multi_gpu import ParallelModelCheckpoint

EPOCHS = 100

G = 2
if G > 1:
    print("[INFO] setting up for multi-gpu")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-w", "--weights", required=True, help="path to weights directory")
ap.add_argument("-b", "--best_only", type=bool, default=True,
                help="If True, lpr_model will only write a single file with best result")
args = vars(ap.parse_args())

# grab the list of images that we?ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images")
imagePaths = []
labels = []
pl = os.walk(args["dataset"])
for root, dirs, files in pl:
    for file in files:
        if '.jpg' in file:
            labels.append(root.split("/")[-1])
            imagePaths.append(root + "/" + file)

classNames = [str(x) for x in np.unique(labels)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(224, 224, gray=True)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessor=[iap], gray=True)
data, labels = sdl.load(imagePaths, verbose=500)
data = data / 255

# # convert the labels from integers to vectors
labels = LabelBinarizer().fit_transform(labels)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')

# convert the labels from integers to vectors
print("[INFO] compiling lpr_model...")
opt = SGD(lr=0.01, decay=0.01 / EPOCHS, momentum=0.9, nesterov=True)

if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = MiniVGGNet.build(width=224, height=224, depth=1, classes=len(classNames))

# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the lpr_model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    single_gpu_model = MicroVGGNet.build(width=224, height=224, depth=1, classes=len(classNames))

    # make the lpr_model parallel
    model = multi_gpu_model(single_gpu_model, gpus=G)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

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

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32 * G), validation_data=(testX, testY),
                        callbacks=callbacks,
                        steps_per_epoch=len(trainX) // (32 * G), class_weight=classWeight, epochs=EPOCHS, verbose=1)

# evaluate the network
print("[INFO] eveluating network...")
predictions = model.predict(testX, batch_size=32 * G)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

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
