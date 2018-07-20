import cv2
import field_test.smart_test.vin_ocr.config.vin_config as config
from tools import paths

# os.environ["DISPLAY"] = "localhost:11.0"  # uncommented it only when you debug through server via X11-forwarding

raw_image_paths = paths.list_images(config.RAW_IMAGE_PATH)

for raw_image_path in raw_image_paths:
    raw_image = cv2.imread(raw_image_path)
    raw_image = cv2.resize(raw_image, (1280,720))
    cv2.imshow('test', raw_image)
    cv2.waitKey(0)
    cv2.sa
