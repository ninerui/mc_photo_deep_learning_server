import tensorflow as tf

from utils import tf_tools


class AIChallengeWithDPEDSRCNN:
    def __init__(self, model_path='./models/image_enhancement_models/ai_challenge_dped_srcnn.pb'):
        graph = tf_tools.load_pb_model(model_path)
        self.sess = tf.Session(graph=graph)
        self.image_tensor = graph.get_tensor_by_name('input:0')
        self.output_tensor = graph.get_tensor_by_name('output:0')

    def get_image(self, images):
        return self.sess.run([self.output_tensor], feed_dict={self.image_tensor: images})
