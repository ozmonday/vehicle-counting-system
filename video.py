import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import utill
import config
import model

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from tools import generate_detections as gdet
from deep_sort.detection import Detection
from matplotlib import pyplot as plt


# mdl = model.YoloV4('assets/class_name.txt', config.cfg, 'assets/weight.h5')
capture = cv.VideoCapture('/home/hadioz/Videos/road-test.mp4')
pts = [(400, 0), (600, 0), (600, 540), (400, 540)]
RED = (0, 0, 255)
#obj_detect = cv.createBackgroundSubtractorMOG2()

class_name = [line.strip() for line in open('assets/class_name.txt').readlines()]
interpreter = tf.lite.Interpreter(model_path='assets/yolo-v4-car-lite-503.130-125.92.tflite')
interpreter.allocate_tensors()

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

matric = NearestNeighborDistanceMetric("cosine", 0.4, None)
tracker = Tracker(matric)

while True:
    # capture frame-by-frame from video file
    ret, frame = capture.read() 
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # mask = obj_detect.apply(frame)
    frame = cv.resize(frame, (960, 540))
    # mask = cv.resize(mask, (960, 540))
    _, boxes = utill.tflite_predict(frame, config.cfg_lite, class_name, interpreter)

    features = encoder(frame, boxes)
  
    dtc = []
    for idx in range(len(boxes)):
        b = [boxes.iloc[idx,0], boxes.iloc[idx,1], boxes.iloc[idx,6], boxes.iloc[idx,7]]
        s = boxes.iloc[idx,5]
        c = boxes.iloc[idx,4]
        f = features[idx]
        dtc.append(Detection(b, s, c, f))
    
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    tracker.predict()
    tracker.update(dtc)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        cn = track.get_class()
            
        # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(cn)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv.putText(frame, cn + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

    result = np.asarray(frame)
    result = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.polylines(frame, np.array([pts]), True, RED, 2)
    cv.imshow("frame", result)
    # # cv.imshow("mask", mask)
    
    if cv.waitKey(27) & 0xFF == ord('q'):
        break
# When everything done, release the capture
capture.release()
cv.destroyAllWindows()