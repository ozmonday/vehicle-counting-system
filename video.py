import numpy as np
import cv2 as cv
import tensorflow as tf
import os
import utill
import config
import model
import counting_line

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from tools import generate_detections as gdet
from deep_sort.detection import Detection
from matplotlib import pyplot as plt


# mdl = model.YoloV4('assets/class_name.txt', config.cfg, 'assets/weight.h5')
capture = cv.VideoCapture('/home/hadioz/Videos/video_test/test-day-1.mp4')
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
print(f'frame_height {frame_height}')
print(f'frame_width {frame_width}')
out = cv.VideoWriter('road-test.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))



p = list(dict(counting_line.test_day_1['area-one']).values())
p = [ (round(x*frame_width), round(y*frame_height)) for x, y in p]
print(p)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

ctx = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv.polylines(ctx, np.array([p]), True, (255, 255, 255), 1)
cnts_zone, _ = cv.findContours(ctx, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

class_name = [line.strip() for line in open('assets/new_class_name.txt').readlines()]
interpreter = tf.lite.Interpreter(model_path='assets/yolo-car-lite-503-final.tflite')
interpreter.allocate_tensors()

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

matric = NearestNeighborDistanceMetric("cosine", 0.4, None)
tracker = Tracker(matric)
heavy_vehicle = 0
light_vehicle = 0
motor_vehicle = 0
unknow_vehicle = 0
pt = utill.PoinTrack()

while True:
    # capture frame-by-frame from video file
    ret, frame = capture.read() 
    if ret == False:
        break
    
    #frame = cv.resize(frame, (960, 540))
    dtc = utill.tflite_predict(frame, config.cfg_lite, class_name, interpreter, encoder)
    
    tracker.predict()
    tracker.update(dtc)
    
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    cout = pt.update()
    for c in cout:
        if c == 'multi-wheeled_vehicle':
            heavy_vehicle += 1
        elif c == 'four-wheeled_vehicle':
            light_vehicle += 1
        elif c == 'two-wheeled_vehicle':
            motor_vehicle += 1
        else:
            unknow_vehicle += 1


    cv.putText(frame, 'heavy_vehicle : ' + str(heavy_vehicle),(50, 50), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)
    cv.putText(frame, 'light_vehicle : ' + str(light_vehicle),(50, 80), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)
    cv.putText(frame, 'motor_vehicle : ' + str(motor_vehicle),(50, 110), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)
    cv.putText(frame, 'unknow_vehicle : ' + str(unknow_vehicle),(50, 140), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)
    cv.polylines(frame, np.array([p]), True, GREEN, 1)
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        center = track.to_tlbr()
        center_x = round((center[0] + center[2])/2)
        center_y = round((center[1] + center[3])/2)
        
        res = cv.pointPolygonTest(cnts_zone[0], (center_x,center_y), True)
        if res > 0:
            pt.check(track.track_id, track.get_class())

        cn = track.get_class()
        # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        text = f'{cn}'

        cv.circle(frame,(center_x, center_y), 5, color, -1) 
        (text_width, text_height) = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, fontScale=0.75, thickness=2)[0]
        cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv.rectangle(frame, (int(bbox[0]), int(bbox[1]-text_height - 10)), (int(bbox[0])+text_width+10, int(bbox[1])), color, -1)       
        cv.putText(frame, text,(int(bbox[0] + 5) , int(bbox[1] - 5)), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)
    
    result = np.asarray(frame)
    out.write(result)
    cv.imshow("frame", result)
    
    if cv.waitKey(27) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
capture.release()

cv.destroyAllWindows()