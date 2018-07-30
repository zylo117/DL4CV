# 加入机制，识别率低于0.90就往复旋转图片5°并计算透视纠正量进行warp并重新测试，√
# 再识别不出来，就CTPN识别文本位置，从而透视纠正，×
# OpenCV color-matching + OTSU + morphology + houghline + warp-perspective纠正车牌变形，√
# 加入逻辑，两个框重复的时候，车牌识别的内容有冲突时，以字数和车牌数相同(7位）的为准，√
# 重叠框取置信度高的框
# SSD检测汽车，再检测车牌，避免干扰
# SSD检测车牌
# 多个车牌检测模型ensemble

import shutil

import cv2
import os
import tensorflow as tf
import numpy as np
import imutils

from field_test.smart_test.license_plate_recognition.detect_license_plate import load_lpr_model, detect_single_img
from tools import maths

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

raw_images_path = '../../../datasets/car_exam/raw_images'
images_path = '../../../datasets/car_exam/images'
confirmed_images_path = '../../../datasets/car_exam/confirmed_images'

KEY = {
    # 'license_plate': ['0111', '0112', '0164', '0322', '0323', '0348', '0351']
    'license_plate': ['0111', '0112', '0164']

}

# init tensorflow session
config = tf.ConfigProto()
# set gpu mem usage ratio
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf_session = tf.Session(config=config)

# load fusion model of license plate recognition
# cascade classifier, finemapping model,,ocr model
lpr_model = load_lpr_model("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")


def predict_license_plate(img_origin, lpr_model,
                          retry_angle=5,
                          confidence_thresh=0.90):
    rotate_anti_clockwise = True
    retry_count = 0
    img_next = img_origin

    CV_fix = False
    while True:
        result, img = detect_single_img(img_next, lpr_model, confidence_thresh=confidence_thresh,
                                        imshow=False,
                                        fine_mapping=True, use_CV_fix=CV_fix)

        # pre-filter
        for r in result:
            # make sure license No. length is 7
            confidence = r[1]
            if confidence > confidence_thresh:
                if len(r[0]) != 7:
                    result.remove(r)

        # remove the overlapping result
        result_com = maths.combination(result, 2)
        for rc in result_com:
            if rc[0][1] > confidence_thresh and rc[1][1] > confidence_thresh:
                coincide, area0, area1 = maths.rect_coincide(rc[0][2], rc[1][2])
                if coincide > 0.6:
                    if area0 < area1:
                        try:
                            result.remove(rc[0])
                        except:
                            pass
                    else:
                        try:
                            result.remove(rc[1])
                        except:
                            pass

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

            if (retry_count > 6 or confidence_max > 0.85) and not CV_fix:
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
            break

    if len(result) > 0:
        result = np.array(result, dtype=object)
        confi = result[:, 1].astype(np.float)
        result = result[np.argmax(confi), 0]
        confi = np.max(confi)
        return result, confi, img
    else:
        return None, 0, None


count = 0
# enter images paths of many cars
pl = os.listdir(raw_images_path)
for p in pl:
    imgs_path = raw_images_path + '/' + p
    if os.path.isdir(imgs_path):
        # enter images path of a single car
        imgs_pl = os.listdir(imgs_path)
        license_no = []
        for img_p in imgs_pl:
            img_path = imgs_path + '/' + img_p

            # loop over every image that satisfies the requirements
            if img_path.lower().endswith(
                    '.jpg') or img_path.lower().endswith(
                '.jpeg') or img_path.lower().endswith(
                '.png'):
                for K in KEY['license_plate']:
                    if K in img_path:
                        count += 1
                        # start processing
                        img = cv2.imread(img_path)
                        # cv2.imshow('test', imutils.resize(img, width=1024))
                        # cv2.waitKey(0)

                        # predict licnese plate
                        result, confidence, img = predict_license_plate(img, lpr_model, retry_angle=5,
                                                                        confidence_thresh=0.90)
                        print(result, confidence)
                        if result is not None and confidence > 0.90 and img is not None:
                            license_no.append(result)

                            output_path = images_path + '/' + result + '/'
                            if not os.path.exists(output_path):
                                os.makedirs(output_path, exist_ok=True)

                            # cv2.imshow('test', imutils.resize(img, width=1024, inter=cv2.INTER_LANCZOS4))
                            # cv2.waitKey(0)
                            cv2.imencode('.jpg', img)[1].tofile(output_path + img_p)

                        print('Image No.{}'.format(count))

        # if all license_no under the same folder matches, consider it's confirmedly labeled
        confirmed = 0
        for i in range(len(license_no)):
            if license_no[i] == license_no[i - 1]:
                confirmed += 1
        if confirmed == len(license_no):
            confirmed = True
        else:
            confirmed = False

        # copy confirmed images to confirmed images path
        confirmed_image_path = confirmed_images_path + '/' + license_no[0] + '/'
        # os.makedirs(confirmed_image_path, exist_ok=True)
        shutil.move(imgs_path, confirmed_image_path)

        if count >= 100:
            raise Exception

print(count)
