import cv2 as cv
import numpy as np
import virtual_zone
import sys
import json
import tensorflow as tf

from tqdm import tqdm
from counting_car import config
from counting_car import utill
from counting_car import sort

from matplotlib import pyplot as plt

capture = cv.VideoCapture(sys.argv[1])

total = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
print('total frame : {}'.format(total))		

frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

with open(sys.argv[2]) as file:
  data = json.load(file)

RoI = [(data['area'][idx], data['area'][idx+1]) for idx in range(0, len(data['area']), 2)]
RoI = [ (round(x*frame_width), round(y*frame_height)) for x, y in RoI]
contex = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv.polylines(contex, np.array([RoI]), True, (255, 255, 255), 1)
v_zone, _ = cv.findContours(contex, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

tracker = sort.Sort(max_age=16, min_hits=1, iou_threshold=0.3)

class_name = [line.strip() for line in open('config/class_name_2.txt').readlines()]
interpreter = tf.lite.Interpreter(model_path='config/model_2.tflite')
interpreter.allocate_tensors()


temp_data = {
    'heavy_vehicle' : 0,
    'light_vehicle' : 0,
    'motor_vehicle' : 0,
    'unknow_vehicle' : 0
}

counter = utill.Counter()

G = (0, 255, 0)
R = (0, 0, 255)

for _ in tqdm(range(total)):
    # capture frame-by-frame from video file
    ret, frame = capture.read() 
    if ret == False:
        break

    boxes = utill.tfl_predict(frame, config.cfg, class_name, interpreter)
    obj = np.zeros((len(boxes), 6))
    for idx in range(len(boxes)):
        obj[idx] = np.array([boxes.iloc[idx,0], boxes.iloc[idx,1], boxes.iloc[idx,2], boxes.iloc[idx,3], boxes.iloc[idx,5], class_name.index(boxes.iloc[idx,4])])
    track = tracker.update(obj)
    
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    
    cout = counter.update()
    for c in cout:
        if c in ['articulated_truck', 'bus', 'single_unit_truck']:
            temp_data['heavy_vehicle'] += 1
        elif c in ['car', 'pickup_truck', 'work_van']:
            temp_data['light_vehicle'] += 1
        elif c in ['motorcycle']:
            temp_data['motor_vehicle'] += 1
        else:
            temp_data['unknow_vehicle'] += 1

    cv.putText(frame, 'heavy_vehicle : ' + str(temp_data['heavy_vehicle']),(50, 50), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)
    cv.putText(frame, 'light_vehicle : ' + str(temp_data['light_vehicle']),(50, 80), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)
    cv.putText(frame, 'motor_vehicle : ' + str(temp_data['motor_vehicle']),(50, 110), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)
    cv.putText(frame, 'unknow_vehicle : ' + str(temp_data['unknow_vehicle']),(50, 140), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255),2)

    fill = []
    for i in range(track.shape[0]):
        color = colors[int(track[i][5]) % len(colors)]
        color = [i * 255 for i in color]
        center = (round(track[i][0] + (track[i][2]-track[i][0])/2), round(track[i][1] + (track[i][3]-track[i][1])/2))
        res = cv.pointPolygonTest(v_zone[0], center, True)
        res = True if res >= 0 else False
        if res:
            counter.check(track[i][5], class_name[int(track[i][4])])
        
        fill.append(res)
        frame = utill.draw_bbox_sort(frame, track[i], class_name, color)
    
    if True in fill:
        cv.polylines(frame, np.array([RoI]), True, R, 2)
    else:
        cv.polylines(frame, np.array([RoI]), True, G, 2)

    out.write(frame)
    frame = cv.resize(frame, (900, 576), interpolation=cv.INTER_NEAREST)
    cv.imshow("frame", frame)
    
    if cv.waitKey(27) & 0xFF == ord('q'):
        break


with open('./output.txt', 'w') as file:
    json.dump(temp_data, file)