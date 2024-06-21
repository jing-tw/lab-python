import cv2 as cv
import glob


from config import Config

def main():
    capture_image()

def capture_image():
    print('Start to capture the image.')
    print('Usage:\n')
    print('\t[s] key to save image.')
    print('\t[Esc] key to exit.')

    
 
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
    
    ok_num = 0
    while cap.isOpened():
        succes, img = cap.read()
        k = cv.waitKey(5)
        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv.imwrite(Config.ImagePath + str('img') + str(num) + '.png', img)
            print("image saved!")
            num += 1

        # check to find the corners
        chessboardSize = (8,6)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret == True:
            ok_num = ok_num + 1
            print('ok_num = ' + str(ok_num) + ', Found the corners')
        # end of check

        cv.imshow('Img',img)


    # Release and destroy all windows before termination
    cap.release()

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()