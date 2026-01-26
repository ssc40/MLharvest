# Python program to identify
#color in images
  
# Importing the libraries OpenCV and numpy
import cv2
import numpy as np
import os
import glob

import glob
from matplotlib import image as mpimg

baseDir = 'rotations-png'
rotDir = os.path.join(baseDir, 'train', '*')
imgs = glob.glob(rotDir)


# n = np.random.randint(len(imgs))


# HAVE TO FIX THIS -- color ranges are still set for red, need to change it for whiter colors.


# fair weather rgb range
lowerFairWeather1 = np.array([240, 240, 240])
upperFairWeather1 = np.array([255, 255, 255])


import glob
import os
import sys
import shutil

data_folder = os.path.join('C:\\', 'Users', 'ssc31_tgwbrje', 'Documents', 'GitHub', 'MLHarvest', 'data', 'earth_img*.png')


def removeCloudsFunc(mydata: str):
    for imgFile in glob.glob(mydata):
        print(imgFile)
        counter = 0
        image = cv2.imread(imgFile)
        result = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fairWeather_mask1 = cv2.inRange(image, lowerFairWeather1, upperFairWeather1)

        result = cv2.bitwise_and(result, result, mask=fairWeather_mask1)
        length = np.shape(fairWeather_mask1)[0] 
        width = np.shape(fairWeather_mask1)[1]
        print(np.shape(fairWeather_mask1))
        counts = np.count_nonzero(fairWeather_mask1)
        percent = 100*counts/(length*width)
        subtitle_string = f'{percent}% of the image is white'
        filename = imgFile.split('\\')[-1]
        if percent > 0.5: #helps to weed out images with too much cloud coverage while not deleting usable images
            counter += 1
            print('To be removed ' + subtitle_string)
            outfile = os.path.join('data', 'Cloudy', os.path.basename(imgFile))
            shutil.copy(imgFile, outfile)
        else:
            print('Not removed ' + subtitle_string)
            outfile = os.path.join('data', 'NotCloudy', os.path.basename(imgFile))
            shutil.copy(imgFile, outfile)

if __name__ == '__main__':
    removeCloudsFunc(data_folder)