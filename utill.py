import numbers
import os
import platform
import numpy as np
import tensorflow as tf
import pandas as pd
import operator
import cv2
import layer

from tensorflow.keras.utils import Sequence
from matplotlib import image
from matplotlib import pyplot as plt
from matplotlib import patches
from tensorflow.keras import layers, backend


class DataGenerator(Sequence):
    def __init__(self, label_anotation, image_path, class_name_path, anchors, target_image_shape=(416, 416, 3),
                 batch_size=64, max_boxes=100, shuffle=True, num_stage=3, bbox_per_grid=3):

        self.label_anotation = label_anotation
        self.image_path = image_path
        self.class_name_path = class_name_path
        self.num_stage = num_stage
        self.bboxs_per_grid = bbox_per_grid
        self.classes = [line.strip() for line in open(class_name_path).readlines()]
        self.max_boxes = max_boxes
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_image_shape = target_image_shape
        self.indexes = np.arange(len(self.label_anotation))
        self.anchors = np.array(anchors).reshape((num_stage*bbox_per_grid, 2))
        self.on_epoch_end()

    def __len__(self):
        '''number of batches per epoch'''
        return int(np.ceil(len(self.label_anotation) / self.batch_size))

    def __getitem__(self, index):

        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data = [self.label_anotation[i] for i in idxs]
        x, y_tensor, y_bbox = self.__data_generation(data)

        return [x, *y_tensor, y_bbox], np.zeros(len(data))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, label_anotation):
        x = np.empty((len(label_anotation), self.target_image_shape[0], self.target_image_shape[1],
                      self.target_image_shape[2]), dtype=np.float32)
        y_bbox = np.empty((len(label_anotation), self.max_boxes, 5), dtype=np.float32)

        for i, line in enumerate(label_anotation):
            img, boxes = get_data(line, image_path=self.image_path, class_name=self.classes, target_image_shape=self.target_image_shape, max_boxes=self.max_boxes)
            x[i] = img
            y_bbox[i] = boxes

        y_tensor, y_true_boxes_xywh = pre_processing_true_bbox(y_bbox, self.target_image_shape[:2], self.anchors,
                                                             len(self.classes), self.num_stage, self.bboxs_per_grid)

        return x, y_tensor, y_true_boxes_xywh


def get_data(data, image_path, class_name, target_image_shape, max_boxes=100):
    if platform.system() == 'Windows':
        name = data['filename'].replace("/", "\\")
    else:
        name = data['filename']
    filepath = os.path.join(image_path, name)
    img = image.imread(filepath) / 255
    if img.shape != target_image_shape:
        img = tf.image.resize(img, (target_image_shape[0], target_image_shape[1]))
    boxes = np.array([[x['x'], x['y'], x['w'], x['h'], class_name.index(x['class'])] for x in data['objects']])
    boxes_data = np.zeros((max_boxes, 5))

    if len(boxes) > 0:
        np.random.shuffle(boxes)
        boxes = boxes[:max_boxes]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * target_image_shape[0]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * target_image_shape[1]
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

        boxes_data[:len(boxes)] = boxes

    return img, boxes_data


def pre_processing_true_bbox(true_boxes, image_size, anchors, num_classes, num_stage, bbox_per_grid):
    anchor_mask = np.arange(0, num_stage*bbox_per_grid, dtype=int)
    anchor_mask = -np.sort(-anchor_mask) # comment this if its worng
    anchor_mask = anchor_mask.reshape((num_stage, bbox_per_grid))
    anchor_mask = anchor_mask.tolist()
    true_boxes = np.array(true_boxes, dtype='float32')
    true_boxes_abs = np.array(true_boxes, dtype='float32')
    image_size = np.array(image_size, dtype='int32')

    true_boxes_xy = (true_boxes_abs[..., 0:2] + true_boxes_abs[..., 2:4]) // 2
    true_boxes_wh = true_boxes_abs[..., 2:4] - true_boxes_abs[..., 0:2]

    true_boxes[..., 0:2] = true_boxes_xy / image_size[::-1]
    true_boxes[..., 2:4] = true_boxes_wh / image_size[::-1]

    bs = true_boxes.shape[0]
    grid_size = [image_size // [8, 16, 32][-(s+1)] for s in range(num_stage)]

    Y_true = [np.zeros((bs, grid_size[s][0], grid_size[s][1], bbox_per_grid, 5 + num_classes), dtype='float32') for s in range(num_stage)]
    Y_true_bbox_xywh = np.concatenate((true_boxes_xy, true_boxes_wh), axis=-1)

    anchors = np.expand_dims(anchors, 0)
    anchors_maxs = anchors / 2.
    anchors_mins = -anchors_maxs
    valid_mask = true_boxes_wh[..., 0] > 0

    for batch_index in range(bs):
        wh = true_boxes_wh[batch_index, valid_mask[batch_index]]
        if len(wh) == 0: continue
        wh = np.expand_dims(wh, -2)

        box_maxs = wh / 2.  # (# of bbox, 1, 2)
        box_mins = -box_maxs

        intersect_mins = np.maximum(box_mins, anchors_mins)
        intersect_maxs = np.minimum(box_maxs, anchors_maxs)
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = np.prod(intersect_wh, axis=-1)
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchors = np.argmax(iou, axis=-1)

        for box_index in range(len(wh)):
            best_anchor = best_anchors[box_index]
            for stage in range(num_stage):
                if best_anchor in anchor_mask[stage]:
                    x_offset = true_boxes[batch_index, box_index, 0] * grid_size[stage][1]
                    y_offset = true_boxes[batch_index, box_index, 1] * grid_size[stage][0]

                    grid_col = np.floor(x_offset).astype('int32')
                    grid_row = np.floor(y_offset).astype('int32')
                    anchor_idx = anchor_mask[stage].index(best_anchor)
                    class_idx = true_boxes[batch_index, box_index, 4].astype('int32')

                    Y_true[stage][batch_index, grid_row, grid_col, anchor_idx, :2] = true_boxes_xy[batch_index,
                                                                                     box_index, :]
                    Y_true[stage][batch_index, grid_row, grid_col, anchor_idx, 2:4] = true_boxes_wh[batch_index,
                                                                                      box_index, :]
                    Y_true[stage][batch_index, grid_row, grid_col, anchor_idx, 4] = 1

                    Y_true[stage][batch_index, grid_row, grid_col, anchor_idx, 5 + class_idx] = 1

    return reversed(Y_true), Y_true_bbox_xywh


def xywh_to_x1y1x2y2(boxes):
    return tf.concat([boxes[..., :2] - boxes[..., 2:] * 0.5, boxes[..., :2] + boxes[..., 2:] * 0.5], axis=-1)


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # w * h
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # (x, y, w, h) -> (x0, y0, x1, y1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    # coordinates of intersection
    top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection_xy = tf.maximum(bottom_right - top_left, 0.0)

    intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]
    union_area = boxes1_area + boxes2_area - intersection_area

    return 1.0 * intersection_area / (union_area + tf.keras.backend.epsilon())


def bbox_giou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # w*h
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # (x, y, w, h) -> (x0, y0, x1, y1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    intersection_xy = tf.maximum(bottom_right - top_left, 0.0)
    intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]

    union_area = boxes1_area + boxes2_area - intersection_area

    iou = 1.0 * intersection_area / (union_area + tf.keras.backend.epsilon())

    enclose_top_left = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_bottom_right = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])

    enclose_xy = enclose_bottom_right - enclose_top_left
    enclose_area = enclose_xy[..., 0] * enclose_xy[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def open_image(path, show = False):
    img = image.imread(path)
    if show:
        plt.figure(figsize = (15,15))
        plt.imshow(img, interpolation='nearest')
        plt.show()
    
    return img


def draw_image(image, notation):
  fig, ax = plt.subplots(figsize = (15,15))
  ax.imshow(image, interpolation='nearest')
  objs = notation["objects"]
  h = notation["image-size"][0]
  w = notation["image-size"][1]
  for obj in objs:
    plt.text(round(obj["x"]* w), round(obj["y"] * h), obj['class'], backgroundcolor='r', color='w', fontweight='bold')
    rect = patches.Rectangle((round(obj["x"]* w), round(obj["y"] * h)), round(obj["w"] * w), round(obj["h"] * h), linewidth=2, edgecolor='r', facecolor='none') 
    ax.add_patch(rect)
  
  plt.show()


def plot_bbox(img, detections, show_img=True):
    """
    Draw bounding boxes on the img.
    :param img: BGR img.
    :param detections: pandas DataFrame containing detections
    :param random_color: assign random color for each objects
    :param cmap: object colormap
    :param plot_img: if plot img with bboxes
    :return: None
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,15))
    ax1.imshow(img, interpolation='nearest')
    ax2.imshow(img, interpolation='nearest')

    for _, row in detections.iterrows():
        x1, y1, x2, y2, cls, score, w, h = row.values
        plt.text(round(x1), round(y1), cls, backgroundcolor='r', color='w', fontweight='bold')
        rect = patches.Rectangle((int(x1), int(y1)), int(w), int(h), linewidth=2, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        
    if show_img:
        plt.show()

def draw_bbox(raw_img, detections, show_text = True):

    raw_img = np.array(raw_img)
    scale = max(raw_img.shape[0:2]) / 416
    line_width = int(1 * scale)

    for _, row in detections.iterrows():
        x1, y1, x2, y2, cls, score, w, h = row.values
        cv2.rectangle(raw_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), line_width)
        if show_text:
            text = f'{cls} {score:.2f}'
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.3 * scale, 0.3)
            thickness = max(int(1 * scale), 1)
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
            cv2.rectangle(raw_img, (x1 - line_width//2, y1 - text_height), (x1 + text_width, y1), (255, 0, 0), cv2.FILLED)
            cv2.putText(raw_img, text, (x1, y1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return raw_img
        



def get_detection_data(model_outputs, img_shape, class_names):
    """
    :param img: target raw image
    :param model_outputs: outputs from inference_model
    :param class_names: list of object class names
    :return:
    """

    num_bboxes = model_outputs[-1][0]
    boxes, scores, classes = [output[0][:num_bboxes] for output in model_outputs[:-1]]

    h = img_shape[0]
    w = img_shape[1]


    df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
    df[['x1', 'x2']] = (df[['x1', 'x2']] * w).astype('int64')
    df[['y1', 'y2']] = (df[['y1', 'y2']] * h).astype('int64')
    if type(classes) != np.ndarray:
        df['class_name'] = np.array(class_names)[classes.numpy().astype('int64')]
    else:
        df['class_name'] = np.array(class_names)[classes.astype('int64')]
    df['score'] = scores
    df['w'] = df['x2'] - df['x1']
    df['h'] = df['y2'] - df['y1']

    return df


def nms(model_ouputs, input_shape, num_class, iou_threshold=0.413, score_threshold=0.3):
    """
    Apply Non-Maximum suppression
    ref: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    :param model_ouputs: yolo model model_ouputs
    :param input_shape: size of input image
    :return: nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
    """
    bs = tf.shape(model_ouputs[0])[0] #beach size
    boxes = tf.zeros((bs, 0, 4))
    confidence = tf.zeros((bs, 0, 1))
    class_probabilities = tf.zeros((bs, 0, num_class))

    for output_idx in range(0, len(model_ouputs), 4):
        output_xy = model_ouputs[output_idx]
        output_conf = model_ouputs[output_idx + 1]
        output_classes = model_ouputs[output_idx + 2]
        boxes = tf.concat([boxes, tf.reshape(output_xy, (bs, -1, 4))], axis=1)
        confidence = tf.concat([confidence, tf.reshape(output_conf, (bs, -1, 1))], axis=1)
        class_probabilities = tf.concat([class_probabilities, tf.reshape(output_classes, (bs, -1, num_class))], axis=1)

    scores = confidence * class_probabilities
    boxes = tf.expand_dims(boxes, axis=-2)
    boxes = boxes / input_shape[0]  # box normalization: relative img size
    #print(f'nms iou: {iou_threshold} score: {score_threshold}')
    (nmsed_boxes,      # [bs, max_detections, 4]
     nmsed_scores,     # [bs, max_detections]
     nmsed_classes,    # [bs, max_detections]
     valid_detections  # [batch_size]
     ) = tf.image.combined_non_max_suppression(
        boxes=boxes,  # y1x1, y2x2 [0~1]
        scores=scores,
        max_output_size_per_class=100,
        max_total_size=100,  # max_boxes: Maximum nmsed_boxes in a single img.
        iou_threshold=iou_threshold,  # iou_threshold: Minimum overlap that counts as a valid detection.
        score_threshold=score_threshold,  # # Minimum confidence that counts as a valid detection.
    )
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections

def read_txt_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    print(sorted_dic_by_value)
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    # if to_show:
    plt.show()
    # close the plot
    # plt.close()


def filtter(boxes, threshold = 0.75):
    index = []
    for idx in range(len(boxes)):
        box_one = np.array([boxes.iloc[idx,0], boxes.iloc[idx,1], boxes.iloc[idx,6], boxes.iloc[idx,7]])
        for idx2 in range(idx+1, len(boxes)):
            box_two = np.array([boxes.iloc[idx2,0], boxes.iloc[idx2,1], boxes.iloc[idx2,6], boxes.iloc[idx2,7]])
            iou = bbox_iou(box_one, box_two)
            if iou >= threshold:
                index.append(idx2)
      
    boxes = boxes.drop(boxes.index[index])
    return boxes

def img_process_tflite(img, shape):
  img_ori = tf.image.resize(img, (shape[1], shape[2]))
  img_ori = img_ori / 255
  img_exp = np.expand_dims(img_ori, axis=0)
  return img_exp

def tflite_predict(img, config, class_name, interpreter, filtter_threshold=0.7):
    anchors = np.array(config['anchors']).reshape((2, 6, 2))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    exp_img = img_process_tflite(img, input_details[0]['shape'])
    interpreter.set_tensor(input_details[0]['index'], exp_img)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(output_details[1]['index']), interpreter.get_tensor(output_details[0]['index'])]
    outputs = layer.yolo_detector_lite(outputs, anchors, len(class_name) , config['strides'], config['xyscale'])
    outputs = nms(outputs, config['image_size'], len(class_name), config['iou_threshold'], config['score_threshold'])
    boxes = get_detection_data(outputs, img.shape, class_name)
    boxes = filtter(boxes, filtter_threshold)
    return draw_bbox(img, boxes)


class Tracker:
    def __init__(self):
        self.leak = []
        self.curent_frame = 0;
        # self.indexs = []

    def check(self, boxes):
        if len(self.leak) == 0:
            for _, row in boxes.iterrows():
                x1, y1, _, _, cls, score, w, h = row.values
                node = {
                    'index': 0, #mengunakan encode jam 
                    'bbox' : [x1, y1, w, h],
                    'class': cls,
                    'confidene' : score,
                    'frame-skip' : 0
                }
                self.leak.append(node)
       # else:

    def clear_leak():
        pass

