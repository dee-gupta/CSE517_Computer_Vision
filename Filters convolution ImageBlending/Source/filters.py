import cv2
import numpy as np


def applyFT(plane):
    ft = np.fft.fft2(plane)
    return np.fft.fftshift(ft)


def applyInverseFT(ftshift):
    fInverseShift = np.fft.ifftshift(ftshift)
    resImage = np.fft.ifft2(fInverseShift)
    return np.abs(resImage)


def applyLowPass(plane):
    ftshift = applyFT(plane)
    rows, cols = plane.shape
    maskRow = rows / 2
    maskCol = cols / 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[maskRow - 10:maskRow + 10, maskCol - 10:maskCol + 10] = 1
    ftshift = ftshift * mask
    return applyInverseFT(ftshift)


def applyHighPass(plane):
    ftshift = applyFT(plane)
    rows, cols = plane.shape
    maskRow = rows / 2
    maskCol = cols / 2
    ftshift[maskRow - 10:maskRow + 10, maskCol - 10:maskCol + 10] = 0
    return applyInverseFT(ftshift)


def getLowPassImage(image):
    colorPlane = cv2.split(image)
    lowPassPlane = list()

    for plane in colorPlane:
        lowPassPlane.append(applyLowPass(plane))

    resultImage = cv2.merge(lowPassPlane)
    return resultImage


def getHighPassImage(image):
    colorPlane = cv2.split(image)
    highPassPlane = list()

    for plane in colorPlane:
        highPassPlane.append(applyHighPass(plane))

    resultImage = cv2.merge(highPassPlane)
    return resultImage
