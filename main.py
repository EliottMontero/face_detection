import cv2
import numpy as np


if __name__ == '__main__':
    mirror = True

    filter2d = False
    gaussian = True
    median = False

    weigthPath = "opencvBasedDetector/detector/res10_300x300_ssd_iter_140000.caffemodel"
    modelPath = "opencvBasedDetector/detector/deploy.prototxt.txt"

    cam = cv2.VideoCapture(0)

    fps = cam.get(cv2.CAP_PROP_FPS) # number of frames per second
    fw = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # frame resolution
    fh = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (fw,fh))

    thr = 0.2
    m = 3
    sigma = 1.5
    changingSigma = False
    changingM = False
    net = cv2.dnn.readNetFromCaffe(modelPath, weigthPath)
    #cam = cv2.VideoCapture('videof.webm')
    while True:
        ret_val, img = cam.read()
        h = img.shape[0]
        w = img.shape[1]
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

        image = img
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        cvDet = net.forward()

        detection = []
        for i in range(0, cvDet.shape[2]): # Iterate over top-200 predicted boxes (defined on the output layer in deploy.prototxt file for caffe network)
            confidence = cvDet[0, 0, i, 2] # For each box, check confidence
            if confidence > thr: # If confidence is over detection_threshold we save the box
                box = cvDet[0, 0, i, 3:7] * np.array([w, h, w, h]) # Get the bounding box from predictions array (see range 3:7 in cvDet[0,0,i,3:7]), returns 4 elements
                (startX, startY, endX, endY) = box.astype("int") # get limits of the bounding box
                text = "{:.2f}%".format(confidence * 100)
                cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(image, text, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                detection.append([ [startX,startY] , [endX,endY] ]) # Append bounding boxes with new format, a list of lists...

        cv2.imshow("OUTPUT", image)

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
