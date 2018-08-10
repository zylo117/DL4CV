import cv2
import json
import os
import numpy as np

from init_tensorflow import init_tensorflow
from field_test.smart_test.license_plate_recognition.deskew import four_perspective
from field_test.smart_test.license_plate_recognition.detect_license_plate import load_lpr_model

raw_images_path = '/home/public/car_exam/raw_images'

KEY = {
    'license_plate': ['0111', '0112', '0164', '0322', '0323', '0348', '0351', '0352']
    # 'license_plate': ['0111', '0112', '0164']

}

# init tensorflow
init_tensorflow()

# load fusion model of license plate recognition
# cascade classifier, finemapping model,,ocr model
lpr_model = load_lpr_model("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")

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
                        json_path = img_path.split('.jpg')[0] + '.json'
                        if os.path.exists(img_path.split('.jpg')[0] + '.json'):
                            count += 1

                            # start processing
                            js = json.loads(open(json_path).read())

                            result = js['results']
                            if len(result) > 0:
                                result = result[0]
                                coords = result['coordinates']
                                coords = [(co['x'], co['y']) for i, co in enumerate(coords)]
                                plate_openalpr = result['plate']

                                img = cv2.imread(img_path)
                                pt1 = (coords[0][0], coords[0][1])
                                pt2 = (coords[1][0], coords[1][1])
                                pt3 = (coords[2][0], coords[2][1])
                                pt4 = (coords[3][0], coords[3][1])

                                # debug
                                # cv2.line(img, pt1, pt2, (255, 0, 255), 2)
                                # cv2.line(img, pt2, pt3, (255, 0, 255), 2)
                                # cv2.line(img, pt3, pt4, (255, 0, 255), 2)
                                # cv2.line(img, pt4, pt1, (255, 0, 255), 2)

                                # extract license plate
                                plate_img = four_perspective(img, pt1, pt2, pt3, pt4)

                                # cv2.imshow('plate_rough', plate_img)
                                # cv2.waitKey(0)
                                plate_img_fm = lpr_model.finemappingVertical_alt(plate_img)

                                # cv2.imshow('test', img)
                                # cv2.waitKey(0)
                                cv2.imshow('plate_fine', np.hstack([plate_img, plate_img_fm]))
                                cv2.waitKey(0)

                                plate_hyperlpr, confidence = lpr_model.recognizeOne(plate_img_fm)

                                msg = 'HyperLPR: {}, OpenALPR: {}'.format(plate_hyperlpr, plate_openalpr)
                                if plate_hyperlpr == plate_openalpr:
                                    print('pass,' + msg + ', ' + img_path.split('.jpg')[0].split('/')[-1])
                                else:
                                    print('fail,' + msg + ', ' + img_path.split('.jpg')[0].split('/')[-1])

                                # print(js)
