from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from starter.knn_first_classifier.img2mem.preprocess import SimplePreprocessor
from starter.knn_first_classifier.img2mem.dataset_loader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
thumb_size = 32
sp = SimplePreprocessor(thumb_size, thumb_size)
sdl = SimpleDatasetLoader([sp])
data, labels = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], thumb_size ** 2 * 3))

# encode the labels as integers
le = LabelEncoder()
le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=5)

# loop over our set of regularizers
for r in (None, "l1", "l2"):
    # train a SGD classifier using a softmax loss function and the
    # specified regularization function for 10 epochs
    print("[INFO] training model with %s penalty" % r)
    model = SGDClassifier(loss="log", penalty=r, max_iter=100, learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] %s penalty accuracy: %.2f" % (r, acc * 100) + "%")
