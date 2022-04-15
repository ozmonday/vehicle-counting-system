import cv2 as cv
import numpy as np
import counting_line
import tensorflow as tf
import config
import utill
import sort


capture = cv.VideoCapture('/home/hadioz/Videos/video_test/test-day-1.mp4')
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
out = cv.VideoWriter('road-test.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))



pov = list(dict(counting_line.test_day_1['area-one']).values())
pov = [ (round(x*frame_width), round(y*frame_height)) for x, y in pov]
contex = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv.polylines(contex, np.array([pov]), True, (255, 255, 255), 1)
view_zone, _ = cv.findContours(contex, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

tracker = sort.Sort(max_age=16, min_hits=1, iou_threshold=0.3)

class_name = [line.strip() for line in open('assets/new_class_name.txt').readlines()]
interpreter = tf.lite.Interpreter(model_path='assets/yolo-car-lite-503-final.tflite')
interpreter.allocate_tensors()


while True:
    # capture frame-by-frame from video file
    ret, frame = capture.read() 
    if ret == False:
        break
    
    frame = cv.resize(frame, (960, 540))
    boxes = utill.tfl_predict(frame, config.cfg_lite, class_name, interpreter)
    obj = np.zeros((len(boxes), 5))
    for idx in range(len(boxes)):
        obj[idx] = np.array([boxes.iloc[idx,0], boxes.iloc[idx,1], boxes.iloc[idx,2], boxes.iloc[idx,3], boxes.iloc[idx,5]])
    track = tracker.update(obj)
    frame = utill.draw_bbox_sort(frame, track)
    
    # out.write(frame)
    cv.imshow("frame", frame)
    
    if cv.waitKey(27) & 0xFF == ord('q'):
        break