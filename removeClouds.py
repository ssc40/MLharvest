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
lowerFairWeather = np.array([240, 240, 240])
upperFairWeather = np.array([255, 255, 255])


import glob
import os
import sys
import shutil

data_folder = os.path.join('C:\\', 'Users', 'ssc31_tgwbrje', 'Documents', 'GitHub', 'MLHarvest', 'DataSet', 'content', 'Data', 'earth_img*.png')


def removeCloudsFunc(mydata: str):
    for imgFile in glob.glob(mydata):
        print(imgFile)
        counter = 0
        image = cv2.imread(imgFile)
    #     print(testImgPath)
        result = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fairWeather_mask = cv2.inRange(image, lowerFairWeather, upperFairWeather)

        result = cv2.bitwise_and(result, result, mask=fairWeather_mask)
        dim = np.shape(fairWeather_mask)[0] 
        counts = np.count_nonzero(fairWeather_mask)
    #     print(counts)
    #     print(dim**2)
        percent = 100*counts/dim**2
        subtitle_string = f'{percent}% of the image is white'
        filename = imgFile.split('\\')[-1]
    #     print(f'{filename}, {subtitle_string}')
        if percent > 10.0:
            counter += 1
            print('To be removed ' + subtitle_string)
            outfile = os.path.join('DataSet', 'content', 'Data', 'Cloudy', os.path.basename(imgFile))
            shutil.copy(imgFile, outfile)
            # path_list.remove(path)
        else:
            print('Not removed ' + subtitle_string)
            outfile = os.path.join('DataSet', 'content', 'Data', 'NotCloudy', os.path.basename(imgFile))
            shutil.copy(imgFile, outfile)

if __name__ == '__main__':
    removeCloudsFunc(data_folder)