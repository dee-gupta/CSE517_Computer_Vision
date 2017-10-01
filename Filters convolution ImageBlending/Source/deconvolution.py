import cv2
import numpy as np


def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im), newsize)
    return np.fft.fftshift(dft)


def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def getDeconvImage(im):
    im = im * 255

    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T
    imf = ft(im, (im.shape[0], im.shape[1]))
    gkf = ft(gk, (im.shape[0], im.shape[1]))
    imconvf = imf / gkf

    resImage = ift(imconvf)
    return resImage
