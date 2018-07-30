import time

import imutils
from keras.backend import set_session
import tensorflow as tf
import cv2
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

from field_test.smart_test.license_plate_recognition.hyperLPR_lite import LPR


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


def detect_single_img(img_origin, lpr_model, confidence_thresh=0.85, imshow=False, fine_mapping=True,
                      use_CV_fix=False):
    """

    :param img_origin: Numpy/Opencv image
    :param lpr_model:
    :param imshow:
    :return:
    """
    # contains pr text, its confidence and its bounding box rect
    result = lpr_model.SimpleRecognizePlateByE2E(img_origin, fine_mapping=fine_mapping, use_CV_fix=use_CV_fix)

    img = img_origin.copy()

    for pstr, confidence, rect in result:
        if confidence > confidence_thresh and len(pstr) == 7:
            img = drawRectBox(img, rect, pstr + " " + str(round(confidence, 3)))
            # print("plate_str:")
            # print(pstr)
            # print("plate_confidence")
            # print(confidence)

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
    # init set gpu mem usage
    # init session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf_session = tf.Session(config=config)

    img_path = "E:/Document/GitHub/DL4CV/datasets/car_exam/raw_images/2018070500283/0111.jpg"
    lpr_model = load_lpr_model("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")

    # 加入机制，识别率低于0.90就往复旋转图片5°并计算透视纠正量进行warp并重新测试，√
    # 再识别不出来，就CTPN识别文本位置，从而透视纠正，×
    # OpenCV color-matching + OTSU + morphology + houghline + warp-perspective纠正车牌变形，√
    # 加入逻辑，两个框重复的时候，车牌识别的内容有冲突时，以字数和车牌数相同的为准
    # SSD检测汽车，再检测车牌，避免干扰
    # SSD检测车牌
    # 检测汽车朝向，对车牌纠正

    img_origin = cv2.imread(img_path)
    rotate_anti_clockwise = True
    retry_count = 0
    img_next = img_origin
    retry_angle = 5
    confidence_thresh = 0.90
    CV_fix = False

    while True:
        result, img = detect_single_img(img_next, lpr_model, confidence_thresh=confidence_thresh,
                                        imshow=False,
                                        fine_mapping=True, use_CV_fix=CV_fix)

        print(result)

        confidence_set = []
        for r in result:
            confidence_set.append(r[1])

        if len(confidence_set) >= 1:
            confidence_max = np.max(np.array(confidence_set))
        else:
            confidence_max = 0

        if confidence_max < confidence_thresh:
            if rotate_anti_clockwise and not CV_fix:
                img_next = imutils.rotate(img_origin, retry_angle * (retry_count + 1))
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
            elif (not rotate_anti_clockwise) and (not CV_fix):
                img_next = imutils.rotate(img_origin, -retry_angle * (retry_count + 1))
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

            if retry_count > 12 and not CV_fix:
                print('rotation fix failed')
                print('try to predict text boundary with CV_fix')
                CV_fix = True
                img_next = img_origin
                continue
            else:
                print('retrying at {}'.format(retry_count))
                retry_count += 1
                if CV_fix:
                    break
        else:
            angle_compensated = retry_angle * np.round(retry_count / 2)
            if retry_count % 2 == 0:
                angle_compensated *= -1
            print('retry counts: {}, angle compensated: {}°'.format(retry_count, angle_compensated))
            img = imutils.resize(img, width=1024, inter=cv2.INTER_LANCZOS4)
            cv2.imshow('test', img)
            cv2.waitKey(0)
            break

    # speed_benchmark(img_path, lpr_model)
