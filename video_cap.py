import cv2 as cv
import numpy as np
import counting_line
import tensorflow as tf
import config
import utill
import sort

from matplotlib import pyplot as plt

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

heavy_vehicle = 0
light_vehicle = 0
motor_vehicle = 0
unknow_vehicle = 0
pt = utill.PoinTrack()

G = (0, 255, 0)
R = (0, 0, 255)

while True:
    # capture frame-by-frame from video file
    ret, frame = capture.read() 
    if ret == False:
        break

    boxes = utill.tfl_predict(frame, config.cfg_lite, class_name, interpreter)
    obj = np.zeros((len(boxes), 6))
    for idx in range(len(boxes)):
        obj[idx] = np.array([boxes.iloc[idx,0], boxes.iloc[idx,1], boxes.iloc[idx,2], boxes.iloc[idx,3], boxes.iloc[idx,5], class_name.index(boxes.iloc[idx,4])])
    track = tracker.update(obj)
    
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

    fill = []
    for i in range(track.shape[0]):
        color = colors[int(track[i][5]) % len(colors)]
        color = [i * 255 for i in color]
        center = (round(track[i][0] + (track[i][2]-track[i][0])/2), round(track[i][1] + (track[i][3]-track[i][1])/2))
        res = cv.pointPolygonTest(view_zone[0], center, True)
        res = True if res >= 0 else False
        if res:
            pt.check(track[i][5], class_name[int(track[i][4])])
        
        fill.append(res)
        frame = utill.draw_bbox_sort(frame, track[i], class_name, color)
    
    if True in fill:
        cv.polylines(frame, np.array([pov]), True, R, 2)
    else:
        cv.polylines(frame, np.array([pov]), True, G, 2)

    out.write(frame)
    cv.imshow("frame", frame)
    
    if cv.waitKey(27) & 0xFF == ord('q'):
        break