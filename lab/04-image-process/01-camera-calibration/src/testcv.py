import cv2 as cv

def main():
    flags = [i for i in dir(cv) if i.startswith('COLOR_')]
    print( flags )

if __name__ == '__main__':
    main()
