import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# initialize the WindowCapture class
wincap = WindowCapture('LDPlayer')

needle_img = cv.imread('To_base.jpg', cv.IMREAD_UNCHANGED)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    result = cv.matchTemplate(screenshot, needle_img, cv.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    threshold = 0.8
    if max_val >= threshold:
        print('Found needle.')

        needle_w = needle_img.shape[1]
        needle_h = needle_img.shape[0]

        top_left = max_loc
        bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

        cv.rectangle(screenshot, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)

        cv.imwrite('result.jpg', screenshot)


    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()            
        break


