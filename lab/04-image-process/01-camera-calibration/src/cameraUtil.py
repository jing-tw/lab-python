import cv2 as cv
import glob

# capture an image
def getCamera():
    # check the camera
    source = 0
    if not glob.glob("/dev/video?"):
        print('No camera detected. Do you plug the camera?')
        return False, None

    cap = cv.VideoCapture(source)
    if cap is None or not cap.isOpened():
       print('Warning: unable to open video source: ', source)
       return False, None
    
    return True, cap
    
