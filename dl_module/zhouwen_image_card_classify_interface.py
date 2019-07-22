import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


class IDCardClassify:
    def __init__(self, model_path='./models/zhouwen_models/idcard_model.pb'):
        self.sess = tf.Session()
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        self.input = self.sess.graph.get_tensor_by_name('id_card_classify/input_1:0')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('id_card_classify/dense_1/Softmax:0')

    def get_res(self, img):
        with self.sess.graph.as_default():
            img = np.expand_dims(np.expand_dims(img, axis=2), axis=0) / 255.0
            pred = self.sess.run(self.softmax_tensor, {self.input: img}).tolist()[0]
            label = pred.index(max(pred))
            confidence = max(pred)
            if label == 0 and confidence > 0.95:
                return 1
            else:
                return 0
