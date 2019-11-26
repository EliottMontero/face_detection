import cv2
import numpy as np


if __name__ == '__main__':
    mirror = True

    filter2d = False
    gaussian = True
    median = False
    
    cam = cv2.VideoCapture(0)

    fps = cam.get(cv2.CAP_PROP_FPS) # number of frames per second
    fw = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame resolution
    fh = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (fw,fh))

    m = 3
    sigma = 1.5
    changingSigma = False
    changingM = False
    #cam = cv2.VideoCapture('videof.webm')
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        if changingSigma:
            sigma = sigma + 0.5
            changingSigma = False
            print("sigma : ", sigma)
        if changingM:
            m = m + 2
            changingM = False
            print("m : ", m)
        cv2.imshow('original', img)

        #FILTER2D
        if(filter2d):
            kernel = np.ones((5,5),np.float32)/25 # create proprietary kernel
            img = cv2.filter2D(img,-1,kernel) # filter with proprietary kernel

        #GAUSSIAN FILTER
        if(gaussian):
            img = cv2.GaussianBlur(img,(m, m), sigma)

        #MEDIAN FILTER
        if(median):
            img = cv2.medianBlur(img,5)

        out.write(img)
        cv2.imshow("Filtered", img)

        k = cv2.waitKey(1)
        if k == 27:
            break  # esc to quit:
        if k == 32:
            changingSigma = True #space
        if k == ord('m'):
            changingM = True
        if k == ord('r'):
            print("RESET, m = 3, sigma = 1.5")
            m = 3
            sigma = 1.5

    out.release()
    print("Output saved in output.avi")
    cv2.destroyAllWindows()
