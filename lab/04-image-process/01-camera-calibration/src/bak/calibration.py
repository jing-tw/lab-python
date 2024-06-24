# Reference: 
#  https://github.com/niconielsen32/CameraCalibration/blob/main/calibration.py
import numpy as np
import cv2 as cv
import glob
import pickle



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


images = sorted(glob.glob('images/*.png'))
print('images', images)

num = 0
for image in images:
    print('start to read image')

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    print('start to find the corners')
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
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
        cv.imwrite('results/img' + str(num) + '.png', img)
        num = num + 1

cv.destroyAllWindows()

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

# UNDISTORTION
img = cv.imread('images/img4.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
print('img w, h = ' + str(w) + ' ' + str(h))

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
cv.imwrite('results/dst.png', dst)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('results/caliResult1.png', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)

h,  w = img.shape[:2]
print('(before map) img w, h = ' + str(w) + ' ' + str(h))
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
cv.imwrite('results/dst_remap.png', dst)
h,  w = dst.shape[:2]
print('dst_remap w, h = ' + str(w) + ' ' + str(h))

# # crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('results/caliResult2.png', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )


### Try to remove the rotation by calibration ###
# img = cv.imread('images/img0.png')

# rotation_vector = rvecs[0]
# #rotation_vector = np.array([rvecs[0][0], rvecs[0][1], rvecs[0][2]])
# print('rotation_vector = ', rotation_vector)
# # Convert rotation vector to rotation matrix
# rotation_matrix, _ = cv.Rodrigues(rotation_vector)

# # Get image dimensions
# height, width = img.shape[:2]

# # Calculate the center of the image
# center_x, center_y = width // 2, height // 2

# # Define the scale factor (1.0 for no scaling)
# scale = 1.0

# # Apply the inverse rotation to the image
# rotated_image = cv.warpAffine(img, d, (width, height), flags=cv.WARP_INVERSE_MAP)
# cv.imwrite('results/rotated_image.png', rotated_image)
# ### end of try ###