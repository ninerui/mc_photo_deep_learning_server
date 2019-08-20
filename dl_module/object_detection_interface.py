import tensorflow as tf

from utils import tf_tools


class ObjectDetectionWithSSDMobilenetV2:
    def __init__(self, model_path='./models/object_detection_ssdlite.pb'):
        detection_graph = tf_tools.load_pb_model(model_path)
        self.sess = tf.Session(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')

    def detect_object(self, images):
        return self.sess.run([self.boxes, self.scores, self.classes], feed_dict={self.image_tensor: images})
