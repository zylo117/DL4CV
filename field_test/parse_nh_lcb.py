import os
import imutils
import cv2
import numpy as np
from imutils import paths

path = '../dataset/LCB_ext/'

pl = os.walk(path)


def remove_sem_marker(img, width=224):
    # remove green marker
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_color = np.dstack([gray, gray, gray])
    diff = img.astype(np.int16) - gray_color.astype(np.int16)
    diff = cv2.convertScaleAbs(diff)
    diff = diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]
    diff[diff < 80] = 0

    if np.count_nonzero(diff) > 0:
        diff = cv2.dilate(diff, None, iterations=2)
        diff = cv2.erode(diff, None, iterations=2)
        dilate = cv2.dilate(diff, None, iterations=4)
        dilate = dilate - diff
        mean = np.mean(gray[dilate > 0])

        gray[diff > 0] = mean

    # remove HUD
    h, w = gray.shape[:2]
    crop_percent = 0.8
    dW = int(w * (1 - crop_percent) / 2)
    dH = int(h * (1 - crop_percent) / 2)

    gray = gray[dH:h - dH, dW:w - dW]

    # crop to main body
    gray = preprocess(gray, 224, 224)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, -16)
    # val, thresh = cv2.threshold(gray,0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    return gray

def preprocess(image, width, height, inter=cv2.INTER_AREA):
    # grab the dimensions of the image and then initialize
    # the deltas to use when cropping
    h, w = image.shape[:2]
    dW = 0
    dH = 0
    # if the width is smaller than the height, then resize
    # along the width (i.e., the smaller dimension) and then
    # update the deltas to crop the height to the desired
    # dimension
    if w < h:
        image = imutils.resize(image, width=width, inter=inter)
        dH = int((image.shape[0] - height) / 2)

    # otherwise, the height is smaller than the width so
    # resize along the height and then update the deltas
    # to crop along the width
    else:
        image = imutils.resize(image, height=height, inter=inter)
        dW = int((image.shape[1] - width) / 2)

    # now that our images have been resized, we need to
    # re-grab the width and height, followed by performing
    # the crop
    h, w = image.shape[:2]
    image = image[dH:h - dH, dW:w - dW]

    # finally, resize the image to the provided spatial
    # dimensions to ensure our output image is always a fixed
    # size
    return cv2.resize(image, (width, height), interpolation=inter)

for root, dirs, files in pl:
    new_root = root.replace("LCB", "LCB_mini")

    for file in files:
        if '.jpg' in file:
            img = cv2.imread(root + '/' + file)
            img = remove_sem_marker(img, width=224)

            os.makedirs(new_root, exist_ok=True)
            cv2.imwrite(new_root + '/' + file, img)
