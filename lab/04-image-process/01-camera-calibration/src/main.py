import numpy as np
import os
import cv2 as cv

from config import Config
from getImage import getImage
from calibration import calibration, undistortion
from pose import poseEstimation, poseEstimation_camera
from drawUtil import DrawOption

def runGetImage():
    print("Get pattern images")
    getImage()
    return True

def runCalibration():
    print("Calibration")
    # do the calibration
    success, cameraMatrix, distCoeff, rvecs, tvecs, objpoints, imgpoints = calibration()
    if success == False:
        print('[Error] calibration faillure.')
        return False
        
    # undistortion the camera and evaluate the error
    undistortion(cameraMatrix, distCoeff, rvecs, tvecs, objpoints, imgpoints)   # for undistortion the camera
    return True

def runPoseEstimate():
    print("Pose estimation")
    # load calibration info 
    root = os.getcwd()
    # cameraMatrix = np.load(os.path.join(root, "cameraMatrix.pkl"), allow_pickle=True)
    # distCoeff = np.load(os.path.join(root, "distCoeff.pkl"), allow_pickle=True)
    (cameraMatrix, distCoeff) = np.load(os.path.join(root, "calibration.pkl"), allow_pickle=True)
    poseEstimation(cameraMatrix, distCoeff, DrawOption.CUBE, nRows=8, nCols=6)  # for calcuating the object pose
    
    return True

def runPoseEstimation_camera():
    print("Pose estimation")
    # load calibration info 
    root = os.getcwd()
    (cameraMatrix, distCoeff) = np.load(os.path.join(root, "calibration.pkl"), allow_pickle=True)
    poseEstimation_camera(cameraMatrix, distCoeff, DrawOption.CUBE, nRows=8, nCols=6)  # for calcuating the object pose
    
    return True

def runAll():
    print("Run all staff.")
    success = runGetImage()
    if not success:
        return False
    success = runCalibration()
    if not success:
        return False
    success = runPoseEstimate()
    if not success:
        return False
    
    return True

def main():
    # Show the menu
    show_menu()

    # Get user's choice
    user_choice = get_user_choice()
    # Process user's choice
    if user_choice == 1:
        success = runGetImage()
    elif user_choice == 2:
        success = runCalibration()
    elif user_choice == 3:
        success = runPoseEstimate()
    elif user_choice == 4:
        success = runPoseEstimation_camera()
    elif user_choice == 5:
        print(" Run all stuff")
        success = runAll()
    elif user_choice == 6:
        print("Exiting the program")
        success = True;
    else:
        print("Invalid choice. Exiting the program.")
        success = False

    return success

def show_menu():
    print("Menu:")
    print("1. Get images.")
    print("2. Calibration.")
    print("3. Pose estimation.")
    print("4. Pose estimation (camera ver.).\n\tInput calibration.pkl. Output: Cubic show")
    print("5. Run all.\n\tInput: None, Output: 1. ./images/img?.png, 2. calibration.pkl, 3. Cubic show")
    print("6. Exit")

def get_user_choice():
    while True:
        try:
            choice = int(input("Enter your choice (1-6): "))
            if 1 <= choice <= 6:
                return choice
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
  

if __name__ == '__main__':
    main()
