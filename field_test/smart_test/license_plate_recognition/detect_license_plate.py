import time

import imutils

from field_test.smart_test.license_plate_recognition.hyperLPR_lite import LPR
from tools.perspective import four_point_transform
import cv2
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


def load_lpr_model(pr_rough_detect_classifier, pr_fine_detect_model, pr_text_ocr_model):
    """
    load license plate recognition lpr_model
    :param pr_rough_detect_classifier:
    :param pr_fine_detect_model:
    :param pr_text_ocr_model:
    :return:
    """
    return LPR(pr_rough_detect_classifier, pr_fine_detect_model, pr_text_ocr_model)


def drawRectBox(image, rect, addText):
    """
    draw rect box on image which show license plate text with chinese supports.
    OpenCV failed to display chinese on image.
    :param image:
    :param rect:
    :param addText:
    :return:
    """
    fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2,
                  cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


def detect_single_img(img_origin, lpr_model, confidence_thresh=0.85, imshow=False):
    """

    :param img_origin: Numpy/Opencv image
    :param lpr_model:
    :param imshow:
    :return:
    """
    # contains pr text, its confidence and its bounding box rect
    result = lpr_model.SimpleRecognizePlateByE2E(img_origin)

    img = img_origin.copy()

    for pstr, confidence, rect in result:
        if confidence > confidence_thresh:
            img = drawRectBox(img, rect, pstr + " " + str(round(confidence, 3)))
            print("plate_str:")
            print(pstr)
            print("plate_confidence")
            print(confidence)

    if imshow:
        cv2.imshow("result", img)
        cv2.waitKey(0)

    return result, img


def speed_benchmark(img_path, lpr_model):
    img = cv2.imread(img_path)
    lpr_model.SimpleRecognizePlateByE2E(img)
    t0 = time.time()
    for x in range(20):
        lpr_model.SimpleRecognizePlateByE2E(img)
    t = (time.time() - t0) / 20.0
    print("Image size: {}x{}, Tact time: {}ms".format(img.shape[1], img.shape[0], round(t * 1000, 2)))


if __name__ == '__main__':
    img_path = "E:/Document/GitHub/DL4CV/datasets/20180705/2018070500283/1111.jpg"
    lpr_model = load_lpr_model("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")

    # 加入机制，识别率低于0.90就自动旋转图片5°重新测试，√
    # SSD检测汽车，再检测车牌，避免干扰
    # SSD检测车牌
    # 检测汽车朝向，对车牌纠正
    # CV纠正车牌变形

    img_origin = cv2.imread(img_path)
    rotate_anti_clockwise = True
    rotate_count = 0
    img_next = img_origin
    retry_angle = 5
    confidence_thresh = 0.75
    while True:
        result, img = detect_single_img(img_next, lpr_model, confidence_thresh=confidence_thresh, imshow=False)

        print(result)

        confidence_set = []
        for r in result:
            confidence_set.append(r[1])

        if len(confidence_set) >= 1:
            confidence_max = np.max(np.array(confidence_set))
        else:
            confidence_max = 0

        if confidence_max < confidence_thresh:
            if rotate_anti_clockwise:
                img_next = imutils.rotate(img_origin, retry_angle * (rotate_count + 1))
                h, w = img_next.shape[:2]
                ori = [[0, 0], [w, 0],
                       [w, h], [0, h]]
                ori = np.array(ori).astype(np.float32)
                dst = [[np.tan(np.deg2rad(retry_angle * 2)) * w, 0], [w, np.tan(np.deg2rad(retry_angle * 2)) * h],
                       [w, h], [0, h]]
                dst = np.array(dst).astype(np.float32)

                # compute the perspective transform matrix and then apply it
                M = cv2.getPerspectiveTransform(ori, dst)
                img_next = cv2.warpPerspective(img_next, M, (img_next.shape[1], img_next.shape[0]))

                # cv2.imshow('car', imutils.resize(img_next, width=1024))
            else:
                img_next = imutils.rotate(img_origin, -retry_angle * (rotate_count + 1))
                h, w = img_next.shape[:2]
                ori = [[0, 0], [w, 0],
                       [w, h], [0, h]]
                ori = np.array(ori).astype(np.float32)
                dst = [[0, np.tan(np.deg2rad(retry_angle * 2)) * h], [w * (1 - np.tan(np.deg2rad(retry_angle * 2))), 0],
                       [w, h], [0, h]]
                dst = np.array(dst).astype(np.float32)

                # compute the perspective transform matrix and then apply it
                M = cv2.getPerspectiveTransform(ori, dst)
                img_next = cv2.warpPerspective(img_next, M, (img_next.shape[1], img_next.shape[0]))

                # cv2.imshow('car', imutils.resize(img_next, width=1024))

            rotate_anti_clockwise = not rotate_anti_clockwise
            rotate_count += 1

            if rotate_count > 12:
                print('rotation fix failed')
                break
            else:
                print('retrying at {}'.format(rotate_count))
        else:
            angle_compensated = retry_angle * np.round(rotate_count / 2)
            if rotate_count % 2 == 0:
                angle_compensated *= -1
            print('retry counts: {}, angle compensated: {}°'.format(rotate_count, angle_compensated))
            img = imutils.resize(img, width=1024, inter=cv2.INTER_LANCZOS4)
            cv2.imshow('test', img)
            cv2.waitKey(0)
            break

    # speed_benchmark(img_path, lpr_model)
