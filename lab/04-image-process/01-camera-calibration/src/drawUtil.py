import numpy as np
import cv2 as cv

class DrawOption:
    AXES = 1
    CUBE = 2

def drawAxes(img, corners, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)
    
    corner = tupleOfInts(corners(0).ravel())
    img = cv.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255,0,0),5)
    img = cv.line(img, corner, tupleOfInts(imgpts[1].ravel()), (255,0,0),5)
    img = cv.line(img, corner, tupleOfInts(imgpts[2].ravel()), (255,0,0),5)

def drawCube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # add green plane
    img = cv.drawContours(img, [imgpts[:4]], -1, (0,255,0), -3)

    # add box borders
    for i in range(4):
        j = i + 4
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
        img = cv.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)
    return img