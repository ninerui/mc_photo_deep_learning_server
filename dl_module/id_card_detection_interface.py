# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils import tf_tools


class IDCardDetection:
    def __init__(self, model_path='./models/detection_models/id_card_detection_model.pb'):
        detection_graph = tf_tools.load_pb_model(model_path)
        self.sess = tf.Session(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def detect_id_card(self, images):
        image_expanded = np.expand_dims(images, axis=0)
        # (boxes, scores, classes, num) = self.sess.run(
        #     [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
        #     feed_dict={self.image_tensor: image_expanded})
        scores, boxes = self.sess.run(
            [self.detection_scores, self.detection_boxes], feed_dict={self.image_tensor: image_expanded})
        if np.max(scores) >= 0.8:
            return boxes[0][0]
            # return True
        else:
            return False
