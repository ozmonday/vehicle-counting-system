import tensorflow as tf
from counting_car.utill import bbox_iou, bbox_giou
import numpy as np

def yolo_postulate(conv_output, anchors, stride, num_class):
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    feature_map_size = conv_shape[1]
    anchor_per_scale = anchors.shape[0] # change able
    conv_output = tf.reshape(conv_output,
                             (batch_size, feature_map_size, feature_map_size, anchor_per_scale, 5 + num_class))

    raw_txty = conv_output[..., 0:2]
    raw_twth = conv_output[..., 2:4]
    raw_conf = conv_output[..., 4:5]
    raw_prob = conv_output[..., 5:]

    y = tf.tile(tf.range(feature_map_size, dtype=tf.int32)[:, tf.newaxis], [1, feature_map_size])
    x = tf.tile(tf.range(feature_map_size, dtype=tf.int32)[tf.newaxis, :], [feature_map_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(
        raw_txty) + xy_grid) * stride  # pengalian terhadap stride membuat titik xy realtif terhadap input image size
    pred_wh = (tf.exp(raw_twth) * anchors)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(raw_conf)
    pred_prob = tf.sigmoid(raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def yolo_loss_layer(conv, pred, label, bboxes, stride, classes, iou_loss_thresh):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_size = pred.shape[3] #bermasalah
    input_size = output_size * stride

    conv = tf.reshape(conv, (batch_size, output_size, output_size, anchor_size, 5 + classes))

    raw_class_prob = conv[..., 5:]
    raw_conf = conv[..., 4:5]

    pred_xywh = pred[..., 0:4]
    pred_conf = pred[..., 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_class_prob = label[:, :, :, :, 5:]
    #center lose
    ciou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)

    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)
    # prob loss
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_class_prob, logits=raw_class_prob)
    # conf loss
    expand_pred_xywh = pred_xywh[:, :, :, :, np.newaxis, :]
    expand_bboxes = bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]

    iou = bbox_iou(expand_pred_xywh, expand_bboxes)
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_thresh, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=raw_conf)
    )

    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return ciou_loss, conf_loss, prob_loss


def yolo_loss(args, classes, iou_loss_thresh, anchors):
    conv_sbbox = args[0]  # (None, 52, 52, 75)
    conv_mbbox = args[1]  # (None, 26, 26, 75)
    conv_lbbox = args[2]  # (None, 13, 13, 75)


    label_sbbox = args[3]  # (None, 52, 52, 3, 25)
    label_mbbox = args[4]  # (None, 26, 26, 3, 25)
    label_lbbox = args[5]  # (None, 13, 13, 3, 25)
    true_boxes = args[6]  # (None, 100, 4)

    pred_sbbox = yolo_postulate(conv_sbbox, anchors[0], 8, classes)  # (None, None, None, 3, 25)
    pred_mbbox = yolo_postulate(conv_mbbox, anchors[1], 16, classes)  # (None, None, None, 3, 25)
    pred_lbbox = yolo_postulate(conv_lbbox, anchors[2], 32, classes)  # (None, None, None, 3, 25)

    sbbox_ciou_loss, sbbox_conf_loss, sbbox_prob_loss = yolo_loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_boxes, 8,
                                                                   classes, iou_loss_thresh)
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss = yolo_loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_boxes, 16,
                                                                   classes, iou_loss_thresh)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss = yolo_loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_boxes, 32,
                                                                   classes, iou_loss_thresh)

    ciou_loss = (lbbox_ciou_loss + sbbox_ciou_loss + mbbox_ciou_loss) * 3.54
    conf_loss = (lbbox_conf_loss + sbbox_conf_loss + mbbox_conf_loss) * 64.3 #change
    prob_loss = (lbbox_prob_loss + sbbox_prob_loss + mbbox_prob_loss) * 1

    return ciou_loss + conf_loss + prob_loss


def yolo_loss_lite(args, classes, iou_loss_thresh, anchors, strides):
 
    conv_mbbox = args[0]  # (None, 26, 26, 75)
    conv_lbbox = args[1]  # (None, 13, 13, 75)

    label_mbbox = args[2]  # (None, 26, 26, 3, 25)
    label_lbbox = args[3]  # (None, 13, 13, 3, 25)
    true_boxes = args[4]  # (None, 100, 4)

    pred_mbbox = yolo_postulate(conv_mbbox, anchors[0], strides[0], classes)  # (None, None, None, 3, 25)
    pred_lbbox = yolo_postulate(conv_lbbox, anchors[1], strides[1], classes)  # (None, None, None, 3, 25)
 
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss = yolo_loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_boxes, 16, classes, iou_loss_thresh)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss = yolo_loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_boxes, 32, classes, iou_loss_thresh)

    ciou_loss = (lbbox_ciou_loss + mbbox_ciou_loss) * 2.54
    conf_loss = (lbbox_conf_loss + mbbox_conf_loss) * 50.3
    prob_loss = (lbbox_prob_loss + mbbox_prob_loss) * 1

    return conf_loss + ciou_loss + prob_loss