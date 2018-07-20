# load the image from disk
import os

import cv2

from field_test.smart_test.find_tripod.config import tripod_config as config
from cnn.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from tools import paths

raw_images_paths = paths.list_images(config.RAW_IMAGE_PATH)

for path in raw_images_paths:
    image = cv2.imread(path)

    aap = AspectAwarePreprocessor(1024, 1024, inter=cv2.INTER_LANCZOS4, gray=False)
    # crop images
    image = aap.preprocess(image)

    dir = config.IMAGE_PATH + '/' + path.split(os.path.sep)[-2] + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    cv2.imwrite(dir + path.split(os.path.sep)[-1], image, [cv2.IMWRITE_JPEG_QUALITY, 100])
