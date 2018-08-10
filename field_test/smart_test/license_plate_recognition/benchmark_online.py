# 加入机制，识别率低于0.90就往复旋转图片5°并计算透视纠正量进行warp并重新测试，√
# 再识别不出来，就CTPN识别文本位置，从而透视纠正，×
# OpenCV color-matching + OTSU + morphology + houghline + warp-perspective纠正车牌变形，√
# 加入逻辑，两个框重复的时候，车牌识别的内容有冲突时，以字数和车牌数相同(7位）的为准，√
# 重叠框取置信度高的框
# SSD检测汽车，再检测车牌，避免干扰
# SSD检测车牌
# 多个车牌检测模型ensemble
import base64
import requests
import shutil

import cv2
import os
import numpy as np
import imutils
import json

# SECRET_KEY = 'FR0M+he5hadow1\'VEc0ME'
# SECRET_KEY = 'sk_b4cecca0bb92c620ffebd893'
# SECRET_KEY = 'sk_d8f4510573cb7b2e1144c239'
SECRET_KEY = 'sk_588d6d08cae2d5568e6a4936'

raw_images_path = '/home/public/car_exam/raw_images'

KEY = {
    'license_plate': ['0111', '0112', '0164', '0322', '0323', '0348', '0351', '0352']
    # 'license_plate': ['0111', '0112', '0164']

}

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
                        if not os.path.exists(img_path.split('.jpg')[0] + '.json'):
                            count += 1
                            # start processing
                            with open(img_path, 'rb') as image_file:
                                img_base64 = base64.b64encode(image_file.read())

                            url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=cn&secret_key=%s' % (
                                SECRET_KEY)
                            r = requests.post(url, data=img_base64)
                            r = r.json()
                            js = json.dumps(r, indent=2)
                            info_file = open(img_path.split('.jpg')[0] + '.json', 'w')
                            info_file.write(js)
                            info_file.close()
                            print('NO.{}, path: {}'.format(count, img_path))

                            if count >= 960:
                                raise Exception
