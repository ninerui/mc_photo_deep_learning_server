import numpy as np
import tensorflow as tf
from PIL import Image, ImageColor


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


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


class ImageLocalColor:
    def __init__(self):
        self.MODEL = DeepLabModel('./models/deeplabv3_pascal_trainval_2018_01_04.pb')

    def get_result(self, input, output):
        original_im = Image.open(input)
        resized_im, seg_map = self.MODEL.run(original_im)
        seg_map = np.where(seg_map == 15, seg_map, 0)
        rgb = ImageColor.getrgb('black')
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
