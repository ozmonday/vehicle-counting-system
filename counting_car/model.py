import os
from matplotlib import pyplot
import numpy as np
from six import iteritems

from tqdm import tqdm
import cv2
import loss
import utill
import tensorflow as tf
from glob import glob
import json
import matplotlib.pyplot as plt

from keras import layers, models, optimizers
from layer import CSPDarkNet53, PANet, yolo_detector


class YoloV4(object):
    def __init__(self, class_name_path, config, weight_path=None):
        super().__init__()
        self.anchor_size = config['anchor_size_perdetector']
        self.anchors = np.array(config['anchors']).reshape((config['detector_count'], config['anchor_size_perdetector'], 2))
        self.image_size = config['image_size']
        self.class_name = [line.strip()
                           for line in open(class_name_path).readlines()]
        self.number_of_class = len(self.class_name)
        self.max_boxes = config['max_boxes']
        self.iou_loss_thresh = config['iou_loss_thresh']
        self.strides = config['strides']
        self.xyscale = config['xyscale']
        self.iou_threshold = config['iou_threshold']
        self.score_threshold = config['score_threshold']
        self.weight_path = weight_path

        self.build_model(load_pretrained=True if self.weight_path else False)

    def build_model(self, load_pretrained=True):
        input_layer = layers.Input(self.image_size)
        backbone = CSPDarkNet53(input_layer)
        output_layer = PANet(backbone, self.number_of_class, self.anchor_size)
        self.yolo_model = models.Model(input_layer, output_layer)

        if load_pretrained:
            self.yolo_model.load_weights(self.weight_path)
            print(f'load from {self.weight_path}')

        y_true = [layers.Input(shape=(output.shape[1], output.shape[2], self.anchor_size, (self.number_of_class + 5))) for output in self.yolo_model.outputs]
        y_true.append(layers.Input(shape=(self.max_boxes, 4)))
        
        loss_list = layers.Lambda(loss.yolo_loss, arguments={
                                  'classes': self.number_of_class, 'iou_loss_thresh': self.iou_loss_thresh, 'anchors': self.anchors})([*self.yolo_model.outputs, *y_true])
        self.training_model = models.Model(
            [self.yolo_model.input, *y_true], loss_list)

        yolo_output = yolo_detector(self.yolo_model.outputs, anchors=self.anchors,
                                         classes=self.number_of_class, strides=self.strides, xyscale=self.xyscale)
        nms = utill.nms(yolo_output, input_shape=self.image_size, num_class=self.number_of_class,
                           iou_threshold=self.iou_threshold, score_threshold=self.score_threshold)
        self.inferance_model = models.Model(input_layer, nms)

        self.training_model.compile(optimizer=optimizers.Adam(
            learning_rate=1e-3), loss=lambda y_true, y_pred: y_pred)
    
    def preprocessing_image(self, img):
        img = img /255
        img = cv2.resize(img, self.image_size[:2])
        return img

    def predict(self, img_path, plot_img=True):
        img_ori = utill.open_image(img_path)
        img = tf.image.resize(img_ori, (self.image_size[0], self.image_size[1]))
        img = img / 255
        img_exp = np.expand_dims(img, axis=0)
        predic = self.inferance_model.predict(img_exp)
        df = utill.get_detection_data(predic, img_ori.shape, self.class_name)
        print(df)
        utill.plot_bbox(img_ori, df, plot_img)

    def predict_raw(self, frame_ori):
        frame = tf.image.resize(frame_ori, self.image_size[:2])
        frame = frame /255
        frame_exp = np.expand_dims(frame, axis=0)
        predic = self.inferance_model(frame_exp)
        df = utill.get_detection_data(predic, frame_ori.shape, self.class_name)
        df = utill.filtter(df, 0.55)
        return utill.draw_bbox(frame_ori, df)
    
    def fit(self, data_train, data_validation, initial_epoch, epochs, callback=None):
        self.training_model.fit(data_train, steps_per_epoch=len(
            data_train), validation_data=data_validation, epochs=epochs, initial_epoch=initial_epoch, callbacks=callback)

    def build_litemodel(self, model_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.yolo_model)
        light_model = converter.convert()
        with open(model_path, 'wb') as f:
            f.write(light_model)
    
    def export_gt(self, image_notation, image_folder_path, export_path):
        for line in image_notation:
            filename = os.path.join(image_folder_path, line['filename'])
            img = line['image-size']
            filename = filename.split(os.sep)[-1].split('.')[0]
            output_path = os.path.join(export_path, filename+'.txt')

            with open(output_path, 'w') as gt_file:
                for object in line['objects']:
                    x1 = round(object['x'] * img[1])
                    y1 = round(object['y'] * img[0])
                    x2 = round(x1 + (object['w'] * img[1]))
                    y2 = round(y1 + (object['h'] * img[0]))
                    gt_file.write('{0} {1} {2} {3} {4}\n'.format(object['class'], x1, y1, x2, y2))
    
    def export_predict(self, image_anotation, image_folder_path, export_path, betch_size=2):
        filenames = [line['filename'] for line in image_anotation]
        img_paths = [os.path.join(image_folder_path, name) for name in filenames]
        for idx in tqdm(range(0, len(img_paths), betch_size)):
            paths = img_paths[idx:idx+2]
            imgs = np.zeros((len(paths), *self.image_size))
            raw_img_shapes = []

            for i, path in enumerate(paths):
                img = utill.open_image(path, False)
                raw_img_shapes.append(img.shape)
                img = self.preprocessing_image(img)
                imgs[i] = img
            
            b_boxes, b_scores, b_classes, b_valid_detections = self.inferance_model.predict(imgs)

            for k in range(len(paths)):
                num_boxes = b_valid_detections[k]
                raw_img_shape = raw_img_shapes[k]
                boxes = b_boxes[k, :num_boxes]
                classes = b_classes[k, :num_boxes]
                scores = b_scores[k, :num_boxes]
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] * raw_img_shape[1])  # w
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] * raw_img_shape[0])  # h
                cls_names = [self.class_name[int(c)] for c in classes]

                ipath = paths[k]
                filename = ipath.split(os.sep)[-1].split('.')[0]
                output_path = os.path.join(export_path, filename+'.txt')
                with open(output_path, 'w') as pred_file:
                    for box_idx in range(num_boxes):
                        b = boxes[box_idx]
                        pred_file.write(f'{cls_names[box_idx]} {scores[box_idx]} {round(b[0])} {round(b[1])} {round(b[2])} {round(b[3])}\n')


    def eval_map(self, gt_folder_path, pred_folder_path, temp_json_folder_path, output_files_path, min_overlap = 0.5):
        """Process Gt"""
        ground_truth_files_list = glob(gt_folder_path + '/*.txt')
        assert len(ground_truth_files_list) > 0, 'no ground truth file'
        ground_truth_files_list.sort()
        # dictionary with counter per class
        gt_counter_per_class = {}
        counter_images_per_class = {}

        gt_files = []
        for txt_file in ground_truth_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # check if there is a correspondent detection-results file
            temp_path = os.path.join(pred_folder_path, (file_id + ".txt"))
            assert os.path.exists(temp_path), "Error. File not found: {}\n".format(temp_path)
            lines_list = utill.read_txt_to_list(txt_file)
            # create ground-truth dictionary
            bounding_boxes = []
            is_difficult = False
            already_seen_classes = []
            for line in lines_list:
                class_name, left, top, right, bottom = line.split()
                # check if class is in the ignore list, if yes skip
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

            # dump bounding_boxes into a ".json" file
            new_temp_file = os.path.join(temp_json_folder_path, file_id+"_ground_truth.json") #TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            gt_files.append(new_temp_file)
            with open(new_temp_file, 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)
        print(gt_classes, gt_counter_per_class)

        """Process prediction"""

        dr_files_list = sorted(glob(os.path.join(pred_folder_path, '*.txt')))

        for class_index, class_name in enumerate(gt_classes):
            bounding_boxes = []
            for txt_file in dr_files_list:
                # the first time it checks if all the corresponding ground-truth files exist
                file_id = txt_file.split(".txt", 1)[0]
                file_id = os.path.basename(os.path.normpath(file_id))
                temp_path = os.path.join(gt_folder_path, (file_id + ".txt"))
                if class_index == 0:
                    if not os.path.exists(temp_path):
                        error_msg = f"Error. File not found: {temp_path}\n"
                        print(error_msg)
                lines = utill.read_txt_to_list(txt_file)
                for line in lines:
                    try:
                        tmp_class_name, confidence, left, top, right, bottom = line.split()
                    except ValueError:
                        error_msg = f"""Error: File {txt_file} in the wrong format.\n 
                                        Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n 
                                        Received: {line} \n"""
                        print(error_msg)
                    if tmp_class_name == class_name:
                        # print("match")
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
            # sort detection-results by decreasing confidence
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
            with open(temp_json_folder_path + "/" + class_name + "_dr.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        """
         Calculate the AP for each class
        """
        sum_AP = 0.0
        ap_dictionary = {}
        # open file to store the output
        with open(output_files_path + "/output.txt", 'w') as output_file:
            output_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}
            for class_index, class_name in enumerate(gt_classes):
                count_true_positives[class_name] = 0
                """
                 Load detection-results of that class
                """
                dr_file = temp_json_folder_path + "/" + class_name + "_dr.json"
                dr_data = json.load(open(dr_file))

                """
                 Assign detection-results to ground-truth objects
                """
                nd = len(dr_data)
                tp = [0] * nd  # creates an array of zeros of size nd
                fp = [0] * nd
                for idx, detection in enumerate(dr_data):
                    file_id = detection["file_id"]
                    gt_file = temp_json_folder_path + "/" + file_id + "_ground_truth.json"
                    ground_truth_data = json.load(open(gt_file))
                    ovmax = -1
                    gt_match = -1
                    # load detected object bounding-box
                    bb = [float(x) for x in detection["bbox"].split()]
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:
                            bbgt = [float(x) for x in obj["bbox"].split()]
                            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                                     (bbgt[2] - bbgt[0]+ 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj

                    if ovmax >= min_overlap:
                        # if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                    else:
                        fp[idx] = 1


                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                print('fp ', cumsum)
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                print('tp ', cumsum)
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                print('recall ', cumsum)
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                print('prec ', cumsum)

                ap, mrec, mprec = utill.voc_ap(rec[:], prec[:])
                sum_AP += ap
                text = "{0:.2f}%".format(
                    ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)

                print(text)
                ap_dictionary[class_name] = ap

                n_images = counter_images_per_class[class_name]
                # lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
                # lamr_dictionary[class_name] = lamr

                """
                 Draw plot
                """
                if True:
                    plt.plot(rec, prec, '-o')
                    # add a new penultimate point to the list (mrec[-2], 0.0)
                    # since the last line segment (and respective area) do not affect the AP value
                    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                    # set window title
                    fig = plt.gcf()  # gcf - get current figure
                    fig.canvas.set_window_title('AP ' + class_name)
                    # set plot title
                    plt.title('class: ' + text)
                    # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                    # set axis titles
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    # optional - set axes
                    axes = plt.gca()  # gca - get current axes
                    axes.set_xlim([0.0, 1.0])
                    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                    # Alternative option -> wait for button to be pressed
                    # while not plt.waitforbuttonpress(): pass # wait for key display
                    # Alternative option -> normal display
                    plt.show()
                    # save the plot
                    # fig.savefig(output_files_path + "/classes/" + class_name + ".png")
                    # plt.cla()  # clear axes for next plot

            # if show_animation:
            #     cv2.destroyAllWindows()

            output_file.write("\n# mAP of all classes\n")
            mAP = sum_AP / n_classes
            text = "mAP = {0:.2f}%".format(mAP * 100)
            output_file.write(text + "\n")
            print(text)

        """
         Count total of detection-results
        """
        # iterate through all the files
        det_counter_per_class = {}
        for txt_file in dr_files_list:
            # get lines to list
            lines_list = utill.read_txt_to_list(txt_file)
            for line in lines_list:
                class_name = line.split()[0]
                # check if class is in the ignore list, if yes skip
                # if class_name in args.ignore:
                #     continue
                # count that object
                if class_name in det_counter_per_class:
                    det_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    det_counter_per_class[class_name] = 1
        # print(det_counter_per_class)
        dr_classes = list(det_counter_per_class.keys())

        """
         Plot the total number of occurences of each class in the ground-truth
        """
        if True:
            window_title = "ground-truth-info"
            plot_title = "ground-truth\n"
            plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
            x_label = "Number of objects per class"
            output_path = output_files_path + "/ground-truth-info.png"
            to_show = False
            plot_color = 'forestgreen'
            utill.draw_plot_func(
                gt_counter_per_class,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                '',
            )

        """
         Finish counting true positives
        """
        for class_name in dr_classes:
            # if class exists in detection-result but not in ground-truth then there are no true positives in that class
            if class_name not in gt_classes:
                count_true_positives[class_name] = 0
        # print(count_true_positives)

        """
         Plot the total number of occurences of each class in the "detection-results" folder
        """
        if True:
            window_title = "detection-results-info"
            # Plot title
            plot_title = "detection-results\n"
            plot_title += "(" + str(len(dr_files_list)) + " files and "
            count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
            plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
            # end Plot title
            x_label = "Number of objects per class"
            output_path = output_files_path + "/detection-results-info.png"
            to_show = False
            plot_color = 'forestgreen'
            true_p_bar = count_true_positives
            utill.draw_plot_func(
                det_counter_per_class,
                len(det_counter_per_class),
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                true_p_bar
            )

        """
         Draw mAP plot (Show AP's of all classes in decreasing order)
        """
        if True:
            window_title = "mAP"
            plot_title = "mAP = {0:.2f}%".format(mAP * 100)
            x_label = "Average Precision"
            output_path = output_files_path + "/mAP.png"
            to_show = True
            plot_color = 'royalblue'
            utill.draw_plot_func(
                ap_dictionary,
                n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                ""
            )

