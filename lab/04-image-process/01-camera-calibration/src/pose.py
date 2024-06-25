
import numpy as np
import glob
import os
import cv2 as cv

from config import Config
from drawUtil import drawAxes, drawCube, DrawOption
from cameraUtil import getCamera

def poseEstimation(cameraMatrix, distCoeff, option: DrawOption, nRows, nCols):
    # Read image
    imgPathList = glob.glob(os.path.join(Config.ImagePath + str('*.png')))

    # Init
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1,2)

    # World points of objects to be drawn
    axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]])
    cubeCorners = np.float32([[0,0,0], [0,3,0],[3,3,0],[3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

    for curImgPath in imgPathList:   
        imgBGR = cv.imread(curImgPath)
        success = drawPose(cameraMatrix, distCoeff, option, nRows, nCols, cubeCorners, axis, termCriteria, worldPtsCur, imgBGR)
        if not success:
            print('[Error] drawPose failure')
            return False
        cv. waitKey(1000)

    return True
        

def poseEstimation_camera(cameraMatrix, distCoeff, option: DrawOption, nRows, nCols):
    # Init
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1,2)

    # World points of objects to be drawn
    axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]])
    cubeCorners = np.float32([[0,0,0], [0,3,0],[3,3,0],[3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])
    
    # init camera
    success, cap = getCamera()
    if not success:
        return False
    
    while cap.isOpened():
        succes, imgBGR = cap.read()
        if not success:
            print('capture image failure')
            return False
        
        # control
        k = cv.waitKey(5)
        if k == 27:
            break

        success = drawPose(cameraMatrix, distCoeff, option, nRows, nCols, cubeCorners, axis, termCriteria, worldPtsCur, imgBGR)
        if not success:
            print('[Error] drawPose failure')
            return False
        
    # Release and destroy all windows before termination
    cap.release()
    cv.destroyAllWindows()
    return True

def drawPose(cameraMatrix, distCoeff, option: DrawOption, nRows, nCols, cubeCorners, axis, termCriteria, worldPtsCur, imgBGR):
    imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
    cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

    if cornersFound == True:                  
        # calcuating the pose
        cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11,11), (-1,-1), termCriteria)
        _, rvecs, tvecs = cv.solvePnP(worldPtsCur, cornersRefined, cameraMatrix, distCoeff)
        print("[debug] solvePnP result, rvecs", rvecs)
        print("[debug] solvePnP result, tvecs", tvecs)

        # draw the pose
        if option == DrawOption.AXES:
            imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, distCoeff)
            imgBGR = drawAxes(imgBGR, cornersRefined, imgpts)

        if option == DrawOption.CUBE:
            imgpts, _ = cv.projectPoints(cubeCorners, rvecs, tvecs, cameraMatrix, distCoeff)
            imgBGR = drawCube(imgBGR, imgpts)

    cv.imshow('Chessboard', imgBGR)
       

    return True