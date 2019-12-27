import tensorflow as tf


def load_pb_model(model_path):
    """
    从pb模型文件加载网络
    :param model_path: pb模型文件的目录
    :return: tensorflow网络图
    """
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            od_graph_def.ParseFromString(fid.read())
            tf.import_graph_def(od_graph_def, name='')
    return model_graph
