import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import utill
import config
import model



mdl = model.YoloV4('assets/class_name.txt', config.cfg, 'assets/weight.h5')
capture = cv.VideoCapture('/home/hadioz/Videos/test-5.mp4')
obj_detect = cv.createBackgroundSubtractorMOG2()

while True:
    # capture frame-by-frame from video file
    ret, frame = capture.read() 
    mask = obj_detect.apply(frame)
    frame = cv.resize(frame, (960, 540))
    frame = mdl.predict_raw(frame)
    mask = cv.resize(mask, (960, 540))
    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    
    if cv.waitKey(27) & 0xFF == ord('q'):
        break
# When everything done, release the capture
capture.release()
cv.destroyAllWindows()