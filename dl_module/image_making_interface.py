import cv2
import numpy as np
import tensorflow as tf

from utils import tf_tools


# def _load_dictionary_with_csv(dict_file):
#     dictionary = dict()
#     with open(dict_file, 'r', encoding='UTF-8') as lines:
#         reader = csv.reader(lines)
#         for idx, item in enumerate(reader):
#             dictionary[str(idx)] = item[2]
#     return dictionary
#
#
# def _load_dictionary_with_csv_11166(dict_file):
#     dictionary = dict()
#     with open(dict_file, 'r', encoding='UTF-8') as lines:
#         reader = csv.reader(lines)
#         for item in reader:
#             dictionary[item[0]] = item[3]
#     return dictionary


def _load_dictionary(dict_file):
    dictionary = dict()
    with open(dict_file, 'r', encoding='utf-8') as lines:
        for line in lines:
            sp = line.strip().split('\t')
            idx, name = sp[0], sp[1]
            dictionary[idx] = name
    return dictionary


def _load_dictionary_1(dict_file):
    dictionary = dict()
    dictionary_1 = dict()
    with open(dict_file, 'r', encoding='utf-8') as lines:
        for line in lines:
            sp = line.strip().split(',')
            dictionary[sp[0]] = sp[1]
            dictionary_1[sp[0]] = sp[2]
    return dictionary, dictionary_1


def preprocess(img):
    raw_h = float(img.shape[0])
    raw_w = float(img.shape[1])
    new_h = 256.0
    new_w = 256.0
    test_crop = 224.0
    if raw_h <= raw_w:
        new_w = (raw_w / raw_h) * new_h
    else:
        new_h = (raw_h / raw_w) * new_w
    img = cv2.resize(img, (int(new_w), int(new_h)))
    img = img[
          int((new_h - test_crop) / 2):int((new_h - test_crop) / 2) + int(test_crop),
          int((new_w - test_crop) / 2):int((new_w - test_crop) / 2) + int(test_crop)
          ]
    img = ((img / 255.0) - 0.5) * 2.0
    img = img[..., ::-1]
    return img


class ImageMakingWithOpenImage:
    def __init__(self, model_path='./models/open_image_graph_5000.pb'):
        oi_5000_graph = tf_tools.load_pb_model(model_path)
        # config = tf.ConfigProto()
        # config.intra_op_parallelism_threads = 44
        # config.inter_op_parallelism_threads = 44
        self.oi_5000_sess = tf.Session(graph=oi_5000_graph)
        self.oi_5000_input = oi_5000_graph.get_tensor_by_name('input_values:0')
        self.oi_5000_prob = oi_5000_graph.get_tensor_by_name('multi_predictions:0')
        self.labels, self.object = _load_dictionary_1("./data/open_image_label_5000.txt")

    def parser_res(self, pred_eval, threshold):
        top_k = pred_eval.argsort()[::-1]
        tag = []
        objects = set()
        is_black_and_white = 0
        for i in top_k:
            # if i == 550:  # 是黑白图
            #     is_black_and_white = 1
            confidence = pred_eval[i]
            if confidence < threshold:
                break
            tag.append({"value": self.labels.get(str(i), ""), "confidence": (int(confidence * 100) + 5000)})
            if self.object.get(str(i), ""):
                objects.add(self.object.get(str(i), ""))
        # return {"tags": tag, "is_black_and_white": is_black_and_white, "classes": list(objects)}
        return {"tags": tag, "classes": list(objects)}

    def get_tag(self, img_path, threshold=0.8):
        input_data = [tf.gfile.FastGFile(i, 'rb').read() for i in img_path]
        predictions_eval = self.oi_5000_sess.run(self.oi_5000_prob, feed_dict={self.oi_5000_input: input_data})
        if len(img_path) == 1:
            return [self.parser_res(predictions_eval, threshold)]
        else:
            all_data = []
            for i in predictions_eval:
                all_data.append(self.parser_res(i, threshold))
            return all_data
        # top_k = predictions_eval.argsort()[::-1]
        # tag = []
        # objects = set()
        # is_black_and_white = 0
        # for i in top_k:
        #     if i == 550:  # 是黑白图
        #         is_black_and_white = 1
        #     confidence = predictions_eval[i]
        #     if confidence < threshold:
        #         break
        #     tag.append({"value": self.labels.get(str(i), ""), "confidence": (int(confidence * 100) + 5000)})
        #     if self.object.get(str(i), ""):
        #         objects.add(self.object.get(str(i), ""))
        # return tag, is_black_and_white, list(objects)


class ImageMakingWithTencent:
    def __init__(self, model_path='./models/tencent_1000.pb', label_path='./data/ml_label_1000.txt'):
        ml_graph = tf_tools.load_pb_model(model_path)
        self.ml_sess = tf.Session(graph=ml_graph)
        self.ml_input = ml_graph.get_tensor_by_name('input:0')
        self.ml_prob = ml_graph.get_tensor_by_name('prob_topk:0')
        self.ml_preb = ml_graph.get_tensor_by_name('preb_topk:0')
        self.labels = _load_dictionary(label_path)

    def get_tag(self, raw_img, threshold=0.5):
        img = preprocess(raw_img)
        probs_topk, preds_topk = self.ml_sess.run(
            [self.ml_prob, self.ml_preb], feed_dict={self.ml_input: np.expand_dims(img, axis=0)})
        tag = []
        probs_topk = np.squeeze(probs_topk)
        preds_topk = np.squeeze(preds_topk)
        for i, pred in enumerate(preds_topk):
            confidence = probs_topk[i]
            if confidence < threshold:
                continue
            tag.append({"value": self.labels.get(str(pred), ""), "confidence": (int(confidence * 100) + 1000)})
        return tag

# class ImageMakingWithTencent1000:
#     def __init__(self, model_path='./models/tencent_1000.pb'):
#         ml_1000_graph = tf_tools.load_pb_model(model_path)
#         self.ml_1000_sess = tf.Session(graph=ml_1000_graph)
#         self.ml_1000_input = ml_1000_graph.get_tensor_by_name('input:0')
#         self.ml_1000_prob = ml_1000_graph.get_tensor_by_name('prob_topk:0')
#         self.ml_1000_preb = ml_1000_graph.get_tensor_by_name('preb_topk:0')
#
#     def image_classify(self, images):
#         return self.ml_1000_sess.run(
#             [self.ml_1000_prob, self.ml_1000_preb], {self.ml_1000_input: np.expand_dims(images, axis=0)})
#
#
# class ImageMakingWithTencent11166:
#     def __init__(self, model_path='./models/tencent_11166.pb'):
#         ml_11166_graph = tf_tools.load_pb_model(model_path)
#         self.ml_11166_sess = tf.Session(graph=ml_11166_graph)
#         self.ml_11166_input = ml_11166_graph.get_tensor_by_name('input:0')
#         self.ml_11166_prob = ml_11166_graph.get_tensor_by_name('prob_topk:0')
#         self.ml_11166_preb = ml_11166_graph.get_tensor_by_name('preb_topk:0')
#
#     def image_classify(self, images):
#         return self.ml_11166_sess.run(
#             [self.ml_11166_prob, self.ml_11166_preb], {self.ml_11166_input: np.expand_dims(images, axis=0)})
