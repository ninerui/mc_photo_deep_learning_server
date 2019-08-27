import cv2
import numpy as np
from tensorflow import keras


def preprocess_input_with_emotion(x, v2=True):
    """
    表情识别数据归一化
    :param x:
    :param v2:
    :return:
    """
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


class FaceEmotionKeras(object):
    def __init__(self, model_path='./models/fer2013_mini_XCEPTION.102-0.66.hdf5'):
        self.emotion_classifier = keras.models.load_model(model_path, compile=False)

    def detection_emotion(self, image):
        """
        人脸表情检测
        :param image: 图片
        :return:
        """
        emotion_target_size = self.emotion_classifier.input_shape[1:3]
        gray_face = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_face = cv2.resize(gray_face, emotion_target_size)
        gray_face = preprocess_input_with_emotion(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(self.emotion_classifier.predict(gray_face))
        return int(emotion_label_arg)
