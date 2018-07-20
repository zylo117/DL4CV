import os


def list_images(imagePath):
    imagePaths = []
    pl = os.walk(imagePath)
    for root, dirs, files in pl:
        for file in files:
            file_lower = file.lower()
            if '.jpg' in file_lower or '.jpeg' in file_lower or '.png' in file_lower or '.bmp' in file_lower or '.tif' in file_lower or '.tiff' in file_lower:
                imagePaths.append(root.replace('\\','/') + "/" + file)

    return imagePaths
