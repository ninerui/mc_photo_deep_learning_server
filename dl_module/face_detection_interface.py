import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

from utils import tf_tools


class FaceDetectionWithSSDMobilenet:
    def __init__(self, model_path='./models/fd_ssd_mobilenet.pb'):
        """
        通过ssd mobilenet进行人脸检测
        :param model_path: pb模型文件目录
        """
        face_detection_graph = tf_tools.load_pb_model(model_path)
        self.sess = tf.Session(graph=face_detection_graph)
        self.image_tensor = face_detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = face_detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = face_detection_graph.get_tensor_by_name('detection_scores:0')

    def detect_face(self, images):
        return self.sess.run([self.boxes, self.scores], feed_dict={self.image_tensor: images})


class FaceDetectionWithMtcnnTF:
    def __init__(
            self, weights_file: str = None, min_face_size: int = 20, steps_threshold: list = None,
            scale_factor: float = 0.709):
        """
        通过mtcnn进行人脸检测
        """
        self.detector = MTCNN(
            weights_file=weights_file, min_face_size=min_face_size, steps_threshold=steps_threshold,
            scale_factor=scale_factor)

    def detect_face(self, image):
        mtcnn_res = self.detector.detect_faces(image)
        nrof_faces = len(mtcnn_res)
        if nrof_faces == 0:
            return None
        img_size = np.asarray(image.shape)[0:2]
        if nrof_faces > 1:
            det = np.asarray([i["box"] for i in mtcnn_res])
            bounding_box_size = (det[:, 2]) * (det[:, 3])
            img_center = img_size / 2
            offsets = np.vstack(
                [(det[:, 0] * 2 + det[:, 2]) / 2 - img_center[1], (det[:, 1] * 2 + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = int(np.argmax(bounding_box_size - offset_dist_squared * 2.0))  # some extra weight on the centering
            return mtcnn_res[bindex]
        return mtcnn_res[0]
