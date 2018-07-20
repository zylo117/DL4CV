from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from cnn.io_.hdf5datasetwriter import HDF5DatasetWriter
from tools import paths
import numpy as np
import progressbar
import argparse
import numpy.random_intel as random  # only available in intel dist python
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
ap.add_argument('-o', '--output', required=True,
                help='path to output HDF5 file')
ap.add_argument('-b', '--batch-size', type=int, default=16,
                help='batch size of images to be passed through network')
ap.add_argument('-s', '--buffer-size', type=int, default=1024,
                help='size of feature extraction buffer')
args = vars(ap.parse_args())

# store the batch size in a convenience variable
bs = args['batch_size']

# grab the list of images that we’ll be describing then randomly
# shuffle them to allow for easy training and testing splits via
# array slicing during training time
print('[INFO] loading images...')
imagePaths = paths.list_images(args['dataset'])
random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the
# labels
labels = [p.split('/')[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the ResNet50 network
print('[INFO] loading network...')
model = InceptionResNetV2(weights='imagenet', include_top=False)

# initialize the HDF5 dataset writer, then store the class label
# names in the dataset
# The final average pooling layer of ResNet50 is 2048-d,
dataset = HDF5DatasetWriter((len(imagePaths), 1536 * 8 * 8),
                            args['output'], dataKey='feature',
                            bufSize=args['buffer_size'], overwrite=True)
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ['Extracting Features: ', progressbar.Percentage(), ' ',
           progressbar.Bar(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
                               widgets=widgets).start()

# loop over the images in batches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(299, 299), interpolation='lanczos')
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as
    # our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the ‘MaxPooling2D‘ outputs
    features = features.reshape((features.shape[0], 1536 * 8 * 8))

    # add the features and labels to our HDF5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()