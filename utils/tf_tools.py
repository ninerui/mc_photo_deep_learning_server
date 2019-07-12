import tensorflow as tf


def load_pb_model(model_path):
    """
    从pb模型文件加载网络
    :param model_path: pb模型文件的目录
    :return: tensorflow网络图
    """
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return model_graph
