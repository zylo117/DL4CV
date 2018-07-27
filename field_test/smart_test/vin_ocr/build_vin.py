import cv2
import os

import imutils
import pytesseract.pytesseract as tesseract

path = 'processed/'
pl = os.listdir(path)
for p in pl:
    if p.endswith('.jpg'):
        img = cv2.imread(path + '/' + p)

        # preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3)
        img = imutils.resize(img, width=72)

        val, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = img.shape[:2]
        img = cv2.copyMakeBorder(img, int(0.2 * h), int(0.2 * h), int(0.2 * w), int(0.2 * w),
                                        cv2.BORDER_CONSTANT, value=0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2, borderType=cv2.BORDER_CONSTANT,
                               borderValue=0)

        img = cv2.medianBlur(img, 3)
        # val, img_p = cv2.threshold(img_p, 1, 255, cv2.THRESH_BINARY)

        # cv2.imshow('test', img_p)
        # cv2.waitKey(0)

        text = tesseract.image_to_string(img, lang='eng')
        print(text)