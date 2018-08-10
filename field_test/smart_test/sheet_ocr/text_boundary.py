import cv2
import imutils
import numpy as np


def text_boundary(gray_img, thresh=None, kernel=None, iterations=8):
    gray_img = imutils.resize(gray_img, width=640)
    # transform img into binary img
    if thresh is None:
        _, img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow('debug1', img)
    cv2.waitKey(0)

    if kernel is None:
        # set kernel size into (1, 3), because line spacing is much bigger than character spacing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    # glue every parts of characters together and separate from one another (vertically)
    img = cv2.dilate(img, kernel, iterations=iterations)
    # glue every parts of characters together and separate from one another for a little bit (horizontally)
    img = cv2.dilate(img, None)
    # revert to normal height
    img = cv2.erode(img, kernel, iterations=iterations)

    cv2.imshow("Character recognition", img)
    cv2.waitKey(0)

    h, w = img.shape[:2]
    text_compact = cv2.resize(img, (w, 1), interpolation=cv2.INTER_AREA)
    text_expand = cv2.resize(text_compact, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imshow('test', text_expand)
    cv2.waitKey(0)

    #
    # cnts = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # character_loc_set = np.zeros((len(cnts), 4))  # create a set that filter the false detected rects
    # for i in range(len(cnts)):
    #     x, y, w, h = cv2.boundingRect(cnts[i])
    #     character_loc_set[i] = [x, y, w, h]
    #
    # # filter those who have less width than the average
    # # rect_mean = np.mean(character_loc_set, axis=0)
    # # rect_map = character_loc_set[:, 2] > rect_mean[2]
    # # character_loc_set = character_loc_set[rect_map]
    # # cnts = np.array(cnts)[rect_map]
    #
    # character_roi_set = np.zeros(len(character_loc_set)).astype(np.ndarray)
    # for i in range(len(character_loc_set)):
    #     x, y, w, h = cv2.boundingRect(cnts[i])
    #     new_x = x + w
    #     new_y = y + h
    #     # draw rect on image
    #     cv2.rectangle(img, (x, y), (new_x, new_y), (255, 0, 255), 1)
    #     character_roi_set[i] = img[y:new_y, x:new_x]
    #
    # return img, character_roi_set, character_loc_set


if __name__ == '__main__':
    img_path = 'f:/temp/test_seg.jpg'
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = text_boundary(img, iterations=2)

    cv2.imshow('test', img)
    cv2.waitKey(0)
