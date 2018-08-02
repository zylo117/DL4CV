import cv2
import imutils
import numpy as np
import mahotas

from field_test.smart_test.license_plate_recognition.kmeans import kMeans
from field_test.smart_test.license_plate_recognition.find_color import car_plate_color


def deskew(ori_img, iteration=36, remove_bg=False):
    # preprocess
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)

    h, w = img.shape[:2]

    val, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.GaussianBlur(img, (7, 7), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iteration, borderType=cv2.BORDER_CONSTANT,
                           borderValue=0)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # rotation fix
    coords = np.column_stack(np.where(img > 0))  # get all non-zero pixel coords
    anchor, size, angle = cv2.minAreaRect(coords)  # bound them with a rotated rect

    # angle of minAreaRect is confusing, recommends to a good answer here
    # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
    if angle < -45:
        angle = -(90 + angle)
        center = (anchor[0] + size[1] / 2, anchor[1] + size[0] / 2)
    else:
        angle = -angle
        center = (anchor[0] + size[0] / 2, anchor[1] + size[1] / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)

    img_padded = cv2.copyMakeBorder(ori_img, int(0.2 * h), int(0.2 * h), int(0.2 * w), int(0.2 * w),
                                    cv2.BORDER_CONSTANT, value=0)

    h_padded, w_padded = img_padded.shape[:2]
    warp = cv2.warpAffine(img_padded, M, (w_padded, h_padded), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if remove_bg:
        mask_padded = cv2.copyMakeBorder(img, int(0.2 * h), int(0.2 * h), int(0.2 * w), int(0.2 * w),
                                         cv2.BORDER_CONSTANT, value=0)
        mask_h_padded, mask_w_padded = mask_padded.shape[:2]
        mask_warp = cv2.warpAffine(mask_padded, M, (mask_w_padded, mask_h_padded), flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        coords = np.column_stack(np.where(mask_warp > 0))  # get all non-zero pixel coords
        anchor, size, angle = cv2.minAreaRect(coords)  # bound them with a rotated rect

        if angle < -45:
            warp = warp[int(anchor[0] - size[1] / 2):int(anchor[0] + size[1] / 2),
                   int(anchor[1] - size[0] / 2):int(anchor[1] + size[0] / 2)]
        else:
            warp = warp[int(anchor[0] - size[0] / 2):int(anchor[0] + size[0] / 2),
                   int(anchor[1] - size[1] / 2):int(anchor[1] + size[1] / 2)]

    return warp


def detrap(ori_img):
    # preprocess
    img_debug = ori_img

    # try to extract car license plate using color matching
    img_color = car_plate_color(img_debug, iteration=12)
    try:
        img_debug = deskew(img_color, remove_bg=True)
        img_debug = imutils.resize(img_debug, width=320, inter=cv2.INTER_LANCZOS4)
        cv2.imshow('img_debug_color', img_debug)
        cv2.waitKey(0)
    except:
        return ori_img

    img = cv2.cvtColor(img_debug, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)

    h, w = img.shape[:2]

    # find out the best otsu despite of black pixels
    if len(img[img > 0]) > 0:
        T = mahotas.otsu(img[img > 0])
    else:
        T = 0

    print('otsu value:{}'.format(T))
    # val, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img[img >= T] = 255
    img[img < T] = 0

    img = cv2.GaussianBlur(img, (13, 13), 0)

    ori_h, ori_w = img.shape[:2]
    img = cv2.copyMakeBorder(img, int(0.2 * ori_h), int(0.2 * ori_h), int(0.2 * ori_w), int(0.2 * ori_w),
                             cv2.BORDER_CONSTANT, value=0)
    img_debug = cv2.copyMakeBorder(img_debug, int(0.2 * ori_h), int(0.2 * ori_h), int(0.2 * ori_w), int(0.2 * ori_w),
                                   cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=32, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    val, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    h, w = img.shape[:2]
    # img = img[int(0.2 * ori_h):h - int(0.2 * ori_h), int(0.2 * ori_w): w - int(0.2 * ori_w)]

    cv2.imshow('img_debug1', img)
    cv2.waitKey(0)

    # remove small contours
    # img = cv2.Canny(img, 1, 1)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        img_cnts = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(img_cnts, [c], 0, (255, 255, 255), -1)
        cv2.imshow('img_cnts', img_cnts)
        cv2.waitKey(0)
        img = img_cnts

    img_edge = cv2.Canny(img, 1, 1)
    img = cv2.bitwise_and(np.dstack([img, img, img]), img_debug)

    h, w = img.shape[:2]
    accum = 10
    lines = cv2.HoughLinesP(img_edge, 1, np.pi / 180, accum, minLineLength=h // 4, maxLineGap=h)

    # gather the adjacent lines together
    if lines is not None and len(lines) >= 4:
        print('found {} lines'.format(len(lines)))
        lines1 = lines[:, 0, :]  # extract to 2d
        k_set = []
        for i, (x1, y1, x2, y2) in enumerate(lines1[:]):
            k_set.append(slope(x1, x2, y1, y2, theta=True))
            # cv2.line(img, (x1, y1), (x2, y2), (128, 128, 128), 5)

        k_set = np.asarray(k_set)
        k_set = k_set.reshape((len(k_set), 1))

        while True:
            # make sure no nan/inf in k
            k_center, k_cluster = kMeans(k_set, 2)
            if k_center is None or k_cluster is None:
                k_center = [0, np.pi / 2]
            k_max = np.max(k_center)
            if not np.isnan(k_max):
                k_center = np.sort(k_center, axis=0)
                break

        h_idx = 0 if np.abs(k_center[0]) < np.abs(k_center[1]) else 1
        v_idx = int(not h_idx)

        theta_h = np.rad2deg(k_center[h_idx])
        theta_v = np.rad2deg(k_center[v_idx]) - theta_h
        if theta_v > 45:
            theta_v = 90 - theta_v
        # fix rotation
        img = imutils.rotate_bound(img, angle=-theta_h)
        cv2.imshow('img_debug_rotation', img)
        cv2.waitKey(0)

        # fix perspective
        ori = [[0, 0], [w - h * np.tan(np.deg2rad(theta_v)), 0],
               [w, h], [h * np.tan(np.deg2rad(theta_v)), h]]
        ori = np.array(ori).astype(np.float32)
        dst = [[0, 0], [w, 0],
               [w, h], [0, h]]
        dst = np.array(dst).astype(np.float32)
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(ori, dst)
        old_h, old_w = img.shape[:2]
        img = cv2.copyMakeBorder(img, int(0.2 * old_h), int(0.2 * old_h), int(0.2 * old_w),
                                       int(0.2 * old_w),
                                       cv2.BORDER_CONSTANT, value=0)

        new_h, new_w = img.shape[:2]
        img = cv2.warpPerspective(img, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        img = deskew(img, remove_bg=True)

    return img


def slope(x1, x2, y1, y2, theta=True):
    if x1 != x2:
        result = (y2 - y1) / (x2 - x1)
    else:
        result = np.inf

    if theta:
        return np.arctan(result)
    else:
        return result


def intercept(x1, x2, y1, y2):
    if x1 != x2:
        return y1 - ((y2 - y1) / (x2 - x1)) * x1
    else:
        return np.inf


def line_intersection(k1, b1, k2, b2):
    # l1 : y = k1 * x + b1
    # l2 : y = k2 * x + b2
    if k1 != k2:
        x = - (b2 - b1) / (k2 - k1)
        y = k1 * x + b1
        return x, y
    else:
        return np.inf, np.inf


if __name__ == '__main__':
    ori_img = cv2.imread('test/img_debug_color_screenshot_02.08.2018.png')
    # ori_img = cv2.imread('f:/temp/test.jpg')

    # warp = deskew(ori_img, remove_bg=True)
    trap = detrap(ori_img)

    cv2.imshow('img_p', trap)
    cv2.waitKey(0)
    # cv2.imshow('warp', warp)
    # cv2.waitKey(0)
    # cv2.imwrite('f:/temp/1231.jpg', warp)
