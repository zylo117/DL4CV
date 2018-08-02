import cv2
import numpy as np
import imutils
from skimage.exposure import exposure


def car_plate_color(img, iteration=1):
    """
    Color matching
    :param img:
    :param color: str, i.e. "green", "red", "blue"
    :return:
    """
    # define the lower and upper boundaries of the "car_plate" in the HSV color space

    # from organge to yellow to green to cyan to blue
    # plate_colorLower = (20, 43, 46)
    # plate_colorUpper = (124, 255, 255)

    # blue
    plate_colorLower = (100, 43, 46)
    plate_colorUpper = (124, 255, 255)

    # white
    white_colorLower = (0, 0, 221)
    white_colorUpper = (180, 30, 255)

    # black
    black_colorLower = (0, 0, 0)
    black_colorUpper = (180, 255, 46)

    for i in range(iteration):

        # rescale
        img = exposure.rescale_intensity(img, out_range=(0, 255))

        # resize the img, blur it, and convert it to the HSV color space
        img = imutils.resize(img, width=300, inter=cv2.INTER_LANCZOS4)
        ori_img = img
        ori_h, ori_w = img.shape[:2]
        img = cv2.copyMakeBorder(img, int(0.2 * ori_h), int(0.2 * ori_h), int(0.2 * ori_w), int(0.2 * ori_w),
                                 cv2.BORDER_CONSTANT, value=0)
        blurred = cv2.GaussianBlur(img, (117, 117), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color, then perform a series of dilations and erosions to remove any small blobs left in the mask
        mask_plate = cv2.inRange(hsv, plate_colorLower, plate_colorUpper)
        # mask_white = cv2.inRange(hsv, white_colorLower, white_colorUpper)
        # mask_black = cv2.inRange(hsv, black_colorLower, black_colorUpper)

        # mask = cv2.bitwise_or(mask_white, mask_plate)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 50))
        mask = cv2.morphologyEx(mask_plate, cv2.MORPH_CLOSE, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) > 0:
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

            mask_cnts = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(mask_cnts, [c], 0, (255, 255, 255), -1)
            mask = mask_cnts

        mask = np.dstack([mask, mask, mask])
        img = cv2.bitwise_and(mask, img)

        h, w = img.shape[:2]
        img = img[int(0.2 * ori_h):h - int(0.2 * ori_h), int(0.2 * ori_w): w - int(0.2 * ori_w)]

    return img


if __name__ == '__main__':
    img = cv2.imread('test/except2.jpg')
    img = car_plate_color(img)
    cv2.imshow('result', img)
    cv2.waitKey(0)
