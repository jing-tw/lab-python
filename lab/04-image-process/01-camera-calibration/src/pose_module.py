# Reference: 
#  https://github.com/niconielsen32/CameraCalibration/blob/main/calibration.py
import numpy as np
import cv2 as cv
import glob
import pickle
import os

from config import Config

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

    
def poseEstimation(camMatrix, distCoeff, option: DrawOption, nRows=8, nCols=6):
    # Retreive calibration parameters 
    # root = os.getcwd()
    # camMatrix = np.load(os.path.join(root, "calibration.pkl"))
    # distCoeff = np.load(os.path.join(root, "dist.pkl"))

    # Read image
    # imgPathList = glob.glob(os.path.join('./images/*.png'))
    imgPathList = glob.glob(os.path.join(Config.ImagePath + str('*.png')))

    # init
    # nRows = 8
    # nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1,2)

    # World points of objects to be drawn
    axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]])
    cubeCorners = np.float32([[0,0,0], [0,3,0],[3,3,0],[3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

    for curImgPath in imgPathList:   
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

        if cornersFound == True:                  
            # calcuating the pose
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11,11), (-1,-1), termCriteria)
            _, rvecs, tvecs = cv.solvePnP(worldPtsCur, cornersRefined, camMatrix, distCoeff)


            # draw the pose
            if option == DrawOption.AXES:
                imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, camMatrix, distCoeff)
                imgBGR = drawAxes(imgBGR, cornersRefined, imgpts)

            if option == DrawOption.CUBE:
                imgpts, _ = cv.projectPoints(cubeCorners, rvecs, tvecs, camMatrix, distCoeff)
                imgBGR = drawCube(imgBGR, imgpts)

            cv.imshow('Chessboard', imgBGR)
            cv. waitKey(1000)


def calibration():
    # Init
    cameraMatrix = None # camera matrix
    dist = None # distortion matrix
    rvecs = None # rotation vector
    tvecs = None # transport vector
    objpoints = None # object location in world view (3D)
    imgpoints = None # image points location in the camera view (2D)

    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
    chessboardSize = (8,6)
    frameSize = (640,480)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 10 #20
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # images = sorted(glob.glob('images/*.png'))
    images = sorted(glob.glob(Config.ImagePath + str('*.png')))
    print('images', images)

    num = 0
    for image in images:
        print('Read image, ', image)

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        print('Find the corners')
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret == False:
            print('[Warning] No corners found. Skip the image.')
            continue

        # If found, add object points, image points (after refining them)
        print('found the corners')
        print('** Number of corners detected:', corners.shape[0])
            
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

        # save the result for checking
        cv.imwrite(Config.ResultPath + str('img') + str(num) + '.png', img)
        num = num + 1

    cv.destroyAllWindows()
    if num == 0:
        print('[Error] There is no suitiable imgpoints for calibration processing')
        return False, cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints
    
    # calibration
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    h = len(rvecs)
    w = len(rvecs[0])
    print('Rotation vectors in (nx3) for each pattern images.\n\t h = ', h, ' w = ', w, ' rvecs = ', rvecs) 
    print('The first entry of rvecs = ', rvecs[0])

    h = len(tvecs)
    w = len(tvecs[0])
    print('Transfer vectors in (nx3) for each pattern images.\n\t h = ', h, ' w = ', w, ' tvecs = ', tvecs) 
    print('The first entry of tvecs = ', tvecs[0])

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
    pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
    pickle.dump(dist, open( "dist.pkl", "wb" ))

    return True, cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints

def undistortion(cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints):
    # UNDISTORTION
    img = cv.imread(Config.ImagePath + str('img4.png'))
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
    print('img w, h = ' + str(w) + ' ' + str(h))

    # Undistort
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    cv.imwrite(Config.ResultPath +str('dst.png'), dst)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(Config.ResultPath + str('caliResult1.png'), dst)

    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)

    h,  w = img.shape[:2]
    print('(before map) img w, h = ' + str(w) + ' ' + str(h))
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    cv.imwrite(Config.ResultPath + str('dst_remap.png'), dst)
    h,  w = dst.shape[:2]
    print('dst_remap w, h = ' + str(w) + ' ' + str(h))

    # # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(Config.ResultPath + str('caliResult2.png'), dst)

    # Reprojection Error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)))

def main():
    success, cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints = calibration()
    if success == False:
        print('[Error] calibration faillure.')
        return False
    undistortion(cameraMatrix, dist, rvecs, tvecs, objpoints, imgpoints)
    poseEstimation(cameraMatrix, dist, DrawOption.CUBE)
    

if __name__ == '__main__':
    main()
