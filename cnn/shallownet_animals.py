from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from starter.cnn.preprocessing.preprocess import ImageToArrayPreprocessor
from starter.cnn.preprocessing.preprocess import SimplePreprocessor
from starter.cnn.datasets.dataset_loader import SimpleDatasetLoader
from starter.cnn.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

__classes__ = 2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessor=[sp, iap])
data, labels = sdl.load(imagePaths, verbose=500)
data = data.astype(np.float) / 255

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
# warning: classes must be >= 3, otherwise fit_transform will privide a different result that leads to error
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

if __classes__ == 2:
    new_trainY = np.zeros((trainY.shape[0], 2)).astype(bool)
    new_trainY[:, 0] = trainY[:, 0] == 0
    new_trainY[:, 1] = new_trainY[:, 0] == False
    trainY = new_trainY
    new_testY = np.zeros((testY.shape[0], 2)).astype(bool)
    new_testY[:, 0] = testY[:, 0] == 0
    new_testY[:, 1] = new_testY[:, 0] == False
    testY = new_testY

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=__classes__)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog"]))

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