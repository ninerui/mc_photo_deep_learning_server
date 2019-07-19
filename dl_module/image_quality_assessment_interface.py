import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.platform import gfile

preprocess_input = keras.applications.mobilenet.preprocess_input


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


class QualityAssessmentModel:
    def __init__(self, model_path='./models/weights_mobilenet_aesthetic_0.07.hdf5'):
        self.base_model = keras.applications.mobilenet
        self.model = self.base_model.MobileNet(
            input_shape=(224, 224, 3), weights=None, include_top=False, pooling='avg')
        x = keras.layers.Dropout(0)(self.model.output)
        x = keras.layers.Dense(units=10, activation='softmax')(x)
        self.nima_model = keras.models.Model(self.model.inputs, x)
        self.nima_model.load_weights(model_path)

    def get_res(self, img):
        res = self.nima_model.predict(
            self.base_model.preprocess_input(np.expand_dims(img, axis=0)), use_multiprocessing=True)
        return calc_mean_score(res)


class AestheticQualityModelWithTF:
    def __init__(self, model_path='./models/mobilenet_aesthetic_0.07.pb'):
        self.sess = tf.Session()
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        self.input = self.sess.graph.get_tensor_by_name('input_1:0')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('dense_1/Softmax:0')

    def get_res(self, img):
        res = self.sess.run(self.softmax_tensor, {self.input: preprocess_input(np.expand_dims(img, axis=0))})
        return calc_mean_score(res)


class TechnicalQualityModelWithTF:
    def __init__(self, model_path='./models/mobilenet_technical_0.11.pb'):
        self.sess = tf.Session()
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        self.input = self.sess.graph.get_tensor_by_name('input_1_1:0')
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('dense_1_1/Softmax:0')

    def get_res(self, img):
        res = self.sess.run(self.softmax_tensor, {self.input: preprocess_input(np.expand_dims(img, axis=0))})
        return calc_mean_score(res)
