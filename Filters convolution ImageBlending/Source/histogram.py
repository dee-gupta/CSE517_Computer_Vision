import cv2
import numpy as np
from copy import deepcopy


def find_hist(img):
    hist, bins = np.histogram(img, bins=256, range=(0, 255))
    return hist


def getCumSum(hist):
    return np.cumsum(hist)


def getEqualizeImage(image):

    colorPlane = cv2.split(image)

    histArray = list()
    cdfArray = list()

    for i in range(len(colorPlane)):
        histArray.append(find_hist(colorPlane[i]))
        cdfArray.append(getCumSum(histArray[i]))

    for i in range(len(cdfArray)):
        cdfArray[i] = cdfArray[i].astype(float)
        cdfArray[i] = cdfArray[i] * 255 / cdfArray[i][-1]
        cdfArray[i] = cdfArray[i].astype(int)

    for i in range(len(colorPlane)):
        for j in range(len(colorPlane[i])):
            for k in range(len(colorPlane[i][0])):
                colorPlane[i][j][k] = cdfArray[i][colorPlane[i][j][k]]

    resultImage = cv2.merge(colorPlane)
    return resultImage
