import tensorflow as tf
from tensorflow.python.framework import graph_util


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(
                '../models/image_classification_models/oidv2-resnet_v1_101.ckpt/oidv2-resnet_v1_101.ckpt.meta')
            saver.restore(
                sess, '../models/image_classification_models/oidv2-resnet_v1_101.ckpt')

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()

            # Freeze the graph def
            output_graph_def = graph_util.convert_variables_to_constants(
                sess, input_graph_def, ['input_values', "multi_predictions"])

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile('../models/image_classification_models/oidv2-resnet_v1_101.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    main()
