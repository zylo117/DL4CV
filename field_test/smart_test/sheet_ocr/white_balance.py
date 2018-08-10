import numpy as np
from skimage.exposure.exposure import rescale_intensity


def white_balance(img):
    width = img.shape[1]
    height = img.shape[0]

    # calculate the mean of the center block
    block_size_R = 100
    block_size_C = 100

    center = [height / 2 - 1, width / 2 - 1]

    center_block = img[int(center[0] - block_size_R / 2 + 1):int(center[0] + int(block_size_R / 2 + 1)),
                   int(center[1] - int(block_size_C / 2) + 1): int(center[1] + int(block_size_C / 2) + 1), :]

    center_mean = np.mean((np.mean(center_block, axis=0)), axis=0)
    balance = np.max(center_mean) / center_mean

    # apply balance gain for each channel
    img = img.astype(np.uint16)
    img = img * balance
    img = rescale_intensity(img, out_range=(0, 255))

    return img.astype(np.uint8)


if __name__ == '__main__':
    import cv2

    img_path = 'F:/temp/test_wb3.jpg'
    img = cv2.imread(img_path)
    img = white_balance(img)
    cv2.imwrite('F:/temp/test_wb_fixed.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
