# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


class IDCardClassify:
    def __init__(self, model_path='./models/zhouwen_models/DenseNet121121.pb'):
        self.sess = tf.Session()
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        self.input = self.sess.graph.get_tensor_by_name('id_card_classify/input_1:0')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('id_card_classify/fc1000/Softmax:0')

    def get_res_from_one(self, img):
        with self.sess.graph.as_default():
            img = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2GRAY)
            # cv2.imwrite('1111111111.jpg', img)
            for i in range(4):
                img_ = np.rot90(img, k=i)
                img_ = [np.expand_dims(img_, axis=2)]
                img_ = np.array(img_) / 255.
                pred = self.sess.run(self.softmax_tensor, {self.input: img_}).tolist()
                pred_ = pred[0]
                # print(pred)
                if pred_[0] >= 0.9:
                    return ['身份证']
                # elif pred_[0] < 0.5:
                #     return []
        return []

    # def get_res(self, img):
    #     with self.sess.graph.as_default():
    #         img = [np.expand_dims(cv2.cvtColor(cv2.resize(i, (64, 64)), cv2.COLOR_BGR2GRAY), axis=2) for i in img]
    #         img = np.array(img) / 255.0
    #         # img = np.expand_dims(np.expand_dims(img, ), axis=0) / 255.0
    #         pred = self.sess.run(self.softmax_tensor, {self.input: img}).tolist()
    #         res_list = []
    #         for i in range(len(pred)):
    #             pred_ = pred[i]
    #             label = pred_.index(max(pred_))
    #             confidence = max(pred_)
    #             if label == 0 and confidence > 0.99:
    #                 res_list.append(['身份证'])
    #             else:
    #                 res_list.append([])
    #     return res_list
