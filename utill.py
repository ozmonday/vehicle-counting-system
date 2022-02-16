import numbers
import os
import platform
import numpy as np
import tensorflow as tf
import pandas as pd
import operator
import cv2

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
