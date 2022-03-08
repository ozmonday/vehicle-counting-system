import numpy as np
import cv2 as cv

capture = cv.VideoCapture('/home/hadioz/Videos/road-test.mp4')
while True:
    # capture frame-by-frame from video file
    ret, frame = capture.read() 
    # show the frame on the screen
    frame = cv.resize(frame, (960, 540))
    cv.imshow("frame", frame)
    
    if cv.waitKey(27) & 0xFF == ord('q'):
        break
# When everything done, release the capture
capture.release()
cv.destroyAllWindows()