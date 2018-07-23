import time

from field_test.smart_test.license_plate_recognition.hyperLPR_lite import LPR
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


def detect_single_img(img_path, lpr_model, imshow=False):
    img_origin = cv2.imread(img_path)

    # contains pr text, its confidence and its bounding box rect
    result = lpr_model.SimpleRecognizePlateByE2E(img_origin)

    for pstr, confidence, rect in result:
        if confidence > 0.3:
            img = drawRectBox(img_origin, rect, pstr + " " + str(round(confidence, 3)))
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
    img_path = "E:/Document/GitHub/DL4CV/datasets/20180705/2018070500283/0321.jpg"
    lpr_model = load_lpr_model("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")

    result = detect_single_img(img_path, lpr_model, imshow=True)

    speed_benchmark(img_path, lpr_model)
