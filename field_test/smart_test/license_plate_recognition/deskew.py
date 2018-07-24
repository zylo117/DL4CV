import cv2
import numpy as np


def deskew(ori_img):
    # preprocess
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)

    h, w = img.shape[:2]

    val, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.GaussianBlur(img, (7, 7), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=36, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    val, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # rotation fix
    coords = np.column_stack(np.where(img > 0))  # get all non-zero pixel coords
    anchor, size, angle = cv2.minAreaRect(coords)  # bound them with a rotated rect

    # angle of minAreaRect is confusing, recommends to a good answer here
    # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    center = (anchor[0] + size[0] / 2, anchor[1] + size[1] / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)

    img_padded = cv2.copyMakeBorder(ori_img, int(0.2 * h), int(0.2 * h), int(0.2 * w), int(0.2 * w),
                                    cv2.BORDER_CONSTANT, value=0)

    h_padded, w_padded = img.shape[:2]
    warp = cv2.warpAffine(img_padded, M, (w_padded, h_padded), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return warp


def detrap(ori_img):
    # preprocess
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)

    h, w = img.shape[:2]

    val, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.GaussianBlur(img, (7, 7), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=32, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    val, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    img = cv2.Canny(img, 1, 1)

    h, w = img.shape[:2]
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 10, minLineLength=h // 4, maxLineGap=h)

    # 下一步合并一定斜率k范围内的直线为一条

    if lines is not None:
        lines1 = lines[:, 0, :]  # extract to 2d
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(ori_img, (x1, y1), (x2, y2),
                     (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 5)

    print('line count: {}'.format(len(lines)))

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imshow('img', ori_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    ori_img = cv2.imread('test/1.jpg')

    # warp = deskew(ori_img)
    trap = detrap(ori_img)

    cv2.imshow('img', ori_img)
    cv2.waitKey(0)
    # cv2.imshow('warp', warp)
    # cv2.waitKey(0)
