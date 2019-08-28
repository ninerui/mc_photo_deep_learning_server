import numpy as np
import tensorflow as tf
from PIL import Image


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = tf.GraphDef()
        with tf.gfile.GFile(tarball_path, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


class ImageLocalColor(object):
    def __init__(self):
        self.MODEL = DeepLabModel('./models/deeplabv3_pascal_trainval_2018_01_04.pb')

    def get_result(self, input, output):
        original_im = Image.open(input)
        resized_im, seg_map = self.MODEL.run(original_im)
        seg_map = np.where(seg_map == 15, seg_map, 0)

        resized_im = original_im
        seg_map = np.array(Image.fromarray(seg_map.astype('uint8')).resize(original_im.size))
        # print(np.expand_dims(seg_map, axis=2).shape)
        # seg_map = cv2.resize(seg_map, (4032, ))

        # rgb = ImageColor.getrgb('black')
        pil_image = resized_im
        pil_image_gray = pil_image.convert('L')
        # pil_image_blur = pil_image.filter(ImageFilter.BLUR)
        # pil_image_gray = pil_image_blur.convert('L')
        mask = seg_map
        # solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(seg_map)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0 * 1. * mask)).convert('L')
        pil_mask_1 = Image.fromarray(np.uint8(255.0 * 1. * mask)).convert('L')
        tmp_img = Image.composite(pil_image, pil_solid_color, pil_mask_1)
        pil_solid_color_1 = Image.fromarray(np.uint8(tmp_img)).convert('RGBA')
        pil_image = Image.composite(pil_solid_color_1, pil_image_gray.convert("RGB"), pil_mask)
        pil_image.convert('RGB').save(output)
