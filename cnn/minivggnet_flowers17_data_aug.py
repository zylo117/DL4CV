from keras.backend import set_session
from keras.utils import multi_gpu_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from cnn.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from cnn.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from cnn.datasets.dataset_loader import SimpleDatasetLoader
from cnn.nn.conv.minivggnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import os

G = 1
if G > 1:
    print("[INFO] setting up for multi-gpu")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessor=[aap, iap])
data, labels = sdl.load(imagePaths, verbose=500)
data = data / 255

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

# convert the labels from integers to vectors
print("[INFO] compiling model...")
opt = SGD(lr=0.05)

if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))

# otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    single_gpu_model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))

    # make the model parallel
    model = multi_gpu_model(single_gpu_model, gpus=G)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=1 * G),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // (1 * G), epochs=100, verbose=1)

# evaluate the network
print("[INFO] eveluating network...")
predictions = model.predict(testX, batch_size=1)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(100), H.history["loss"], label="train_loss")
plt.plot(np.arange(100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(100), H.history["acc"], label="train_acc")
plt.plot(np.arange(100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
