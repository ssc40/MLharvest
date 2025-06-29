# Python program to identify
#color in images
  
# Importing the libraries OpenCV and numpy
import cv2
import numpy as np
import os
import glob

import glob
from matplotlib import image as mpimg
%matplotlib inline

baseDir = 'rotations-png'
rotDir = os.path.join(baseDir, 'train', '*')
imgs = glob.glob(rotDir)


# n = np.random.randint(len(imgs))


# HAVE TO FIX THIS -- color ranges are still set for red, need to change it for whiter colors.


# lower boundary RED color range values; Hue (0 - 10)
lower1 = np.array([0, 80, 20])
upper1 = np.array([10, 255, 255])
 
# upper boundary RED color range values; Hue (160 - 180)
lower2 = np.array([160, 100, 20])
upper2 = np.array([179, 255, 255])

counter = 0
for n in range(len(imgs)):
    testImgPath = imgs[n]
    image = cv2.imread(testImgPath)
#     print(testImgPath)
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)
    full_mask = lower_mask + upper_mask;

    result = cv2.bitwise_and(result, result, mask=full_mask)
    dim = np.shape(full_mask)[0] 
    counts = np.count_nonzero(full_mask)
#     print(counts)
#     print(dim**2)
    percent = 100*counts/dim**2
    subtitle_string = f'{percent}% of the image is red'
    filename = testImgPath.split('\\')[-1]
#     print(f'{filename}, {subtitle_string}')
    if percent > 50.0:
        counter += 1
#         print('To be removed')
    else:
        pass


print(counter)

