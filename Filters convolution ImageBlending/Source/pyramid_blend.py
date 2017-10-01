import numpy as np
import cv2


def pyrBlend(imageA, imageB):

    imageA = imageA[:, :imageA.shape[0]]
    imageB = imageB[:imageA.shape[0], :imageA.shape[0]]

    N = 5

    copyA = imageA.copy()
    gListA = [copyA]

    for i in range(N - 1):
        copyA = cv2.pyrDown(copyA)
        gListA.append(copyA)

    copyB = imageB.copy()
    gListB = [copyB]

    for i in range(N - 1):
        copyB = cv2.pyrDown(copyB)
        gListB.append(copyB)

    lapListA = [gListA[N - 1]]

    for i in range(N - 2, -1, -1):
        lapListA.append(cv2.subtract(gListA[i], cv2.pyrUp(gListA[i + 1])))

    lapListB = [gListB[N - 1]]

    for i in range(N - 2, -1, -1):
        lapListB.append(cv2.subtract(gListB[i], cv2.pyrUp(gListB[i + 1])))

    mergeList = []

    for i in range(N):
        mergeList.append(
            np.hstack((lapListA[i][:, :lapListA[i].shape[1] / 2], lapListB[i]
                       [:, lapListB[i].shape[1] / 2:])))

    res = mergeList[0]
    for i in range(1, N):
        res = cv2.pyrUp(res)
        res = cv2.add(res, mergeList[i])

    return res
