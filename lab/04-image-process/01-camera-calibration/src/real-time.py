# real-time 
import cv2 as cv
import numpy as np
import glob

from config import Config


def main():
    print('Start to capture the image.')
    print('Usage:\n')
    print('\t[s] key to save image.')
    print('\t[Esc] key to exit.')

    run()

def run():
    chessboardSize = (8,6)
    frameSize = (640,480)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 10 #20
    objp = objp * size_of_chessboard_squares_mm

    # capture an image
    # check the camera
    source = 0
    if not glob.glob("/dev/video?"):
        print('No camera detected. Do you plug to the camera?')
        return False

    cap = cv.VideoCapture(source)
    if cap is None or not cap.isOpened():
       print('Warning: unable to open video source: ', source)
       return False
    
    num = 0
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ok_num = 0
    while cap.isOpened():
        succes, img = cap.read()
       
        # check to find the corners
        chessboardSize = (8,6)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret == True:
            ok_num = ok_num + 1
            print('ok_num = ' + str(ok_num) + ', Found the corners')

            objpoints.append(objp)
            # draw the corner
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        # end of check

        # control
        k = cv.waitKey(5)
        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv.imwrite(Config.ImagePath + str('img') + str(num) + '.png', img)
            print("image saved!")
            num += 1
        # end of control

        cv.imshow('Img',img)


    # Release and destroy all windows before termination
    cap.release()

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()