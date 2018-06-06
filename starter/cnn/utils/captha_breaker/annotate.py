from imutils import paths
import argparse
import imutils
import cv2
import os
from pytesseract import pytesseract

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the dicectory of images")
ap.add_argument("-a", "--annotate", required=True, help="path to output directory of annotations")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # display an update to the user
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

    try:
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image
        # are retained
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image to reveal the digits
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # find contours in the image, keeping only the four largest
        # ones
        _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        # loop over the contours
        for c in cnts:
            # compute the bounding box for the contour then extract
            # the digit
            # the digit
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

            # display the character, making it larger enough for us
            # to see, then wait for a keypress
            cv2.imshow("ROI", imutils.resize(roi, width=28))

            # use tesseract to OCR the number
            key = pytesseract.image_to_string(roi, lang='eng', config='-psm 10 digits')
            print(key)
            key = int(key)
            # if the "'" key is pressed, then ignore the character
            if key == ord("'"):
                print("[INFO] ignoring character")
                continue

            # grab the key that was pressed and construct the path
            # the output directory
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annotate"], key])
            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # write the labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.jpg".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            # increment the count for the current key
            counts[key] = count + 1

    # we are trying to control-c out of the script, so break from the
    # loop (you still need to press a key for the active window to
    # trigger this)
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break

    # an unknown error has occurred for this particular image
    except:
        print("[INFO] skipping image...")

