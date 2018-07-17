import os

import cv2
import field_test.vin_ocr.config .vin_config as config
from tools import paths

os.environ["DISPLAY"] = "localhost:11.0"

raw_image_paths = paths.list_images(config.RAW_IMAGE_PATH)

for raw_image_path in raw_image_paths:
    raw_image = cv2.imread(raw_image_path)
    raw_image = cv2.resize()
    cv2.imshow('test', raw_image)
    cv2.waitKey(0)