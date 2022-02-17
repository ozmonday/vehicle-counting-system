cfg = {
  'image_size' : (416, 416, 3),
  'anchors' : [4, 7, 9, 17, 17, 40, 31, 51, 49, 83, 82, 125, 98, 212, 175, 169, 194, 311],
  'strides' : [8, 16, 32],
  'xyscale': [1.2, 1.1, 1.05],

  # Training
  'iou_loss_thresh': 0.5,
  'batch_size': 8,
  'num_gpu': 1,  # 2,

  # Inference
  'max_boxes': 100,
  'iou_threshold': 0.413,
  'score_threshold': 0.3,
}