cfg = {
  'image_size' : (416, 416, 3),
  'anchors' : [16, 16, 32, 23, 25, 40, 53, 41, 46, 74, 79, 61, 106, 78, 84, 110, 136, 99, 118, 145, 174, 131, 163, 200, 248, 170, 229, 281, 344, 230],
  'strides' : [8, 16, 32],
  'xyscale': [1.2, 1.1, 1.05],
  'detector_count' : 3,
  'anchor_size_perdetector': 5,

  # Training
  'iou_loss_thresh': 0.5,
  'batch_size': 8,
  'num_gpu': 1,  # 2,

  # Inference
  'max_boxes': 100,
  'iou_threshold': 0.413,
  'score_threshold': 0.5,
}