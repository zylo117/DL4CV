import os


def list_images(imagePath):
    imagePaths = []
    pl = os.walk(imagePath)
    for root, dirs, files in pl:
        for file in files:
            if '.jpg' in file or '.jpeg' in file or '.png' in file or '.bmp' in file or '.tif' in file or '.tiff' in file:
                imagePaths.append(root + "/" + file)

    return imagePaths
