import numpy as np
import cv2
import os

from tools.pic_proc.fix_jpeg import gif2jpeg

class SimpleDatasetLoader:
    def __init__(self, preprocessor=None, gray=False):
        self.preprocessors = preprocessor
        self.gray = gray

        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath) if not self.gray else cv2.imread(imagePath, 0)

            # if image is gif, opencv won't be able to read it due to the patent issue.
            if image is None:
                fixed = gif2jpeg(os.path.abspath(imagePath))  # convert gif(fake jpeg) to real jpeg
                if fixed:
                    image = cv2.imread(imagePath)
                else:
                    continue

            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every "verbose" images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed %d/%d" % (i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))
