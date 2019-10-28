import tensorflow as tf

from utils import tf_tools


class ImageMakingWithOpenImage(object):
    def __init__(self, model_path='./models/image_classification_models/oidv2-resnet_v1_101.pb'):
        oi_5000_graph = tf_tools.load_pb_model(model_path)
        self.oi_5000_sess = tf.Session(graph=oi_5000_graph)
        self.oi_5000_input = oi_5000_graph.get_tensor_by_name('input_values:0')
        self.oi_5000_prob = oi_5000_graph.get_tensor_by_name('multi_predictions:0')

    def get_tag_from_one(self, img_path, threshold=0.8, debug=False):
        predictions_eval = self.oi_5000_sess.run(
            self.oi_5000_prob,
            feed_dict={self.oi_5000_input: [tf.gfile.FastGFile(img_path, 'rb').read()]})
        top_k = predictions_eval.argsort()[::-1]
        tag = []
        for i in top_k:
            confidence = predictions_eval[i]
            if debug:
                print(i, confidence)
            if confidence < threshold:
                break
            tag.append(i + 1)
        return tag
