import tensorflow as tf

from utils import tf_tools
import numpy as np


# 'num_detections', 'detection_boxes', 'detection_scores',
# 'detection_classes', 'detection_masks'

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
    """Transforms the box masks back to full image masks.
    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.
    Args:
      box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.
    Returns:
      A tf.float32 tensor of size [num_masks, image_height, image_width].
    """

    # TODO(rathodv): Make this a public function.
    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""

        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
        return tf.image.crop_and_resize(
            image=box_masks_expanded,
            boxes=reverse_boxes,
            box_ind=tf.range(num_boxes),
            crop_size=[image_height, image_width],
            extrapolation_value=0.0)

    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
    return tf.squeeze(image_masks, axis=3)


class ObjectMaskDetection:
    def __init__(self, model_path='./models/mask_detection_models.pb'):
        detection_graph = tf_tools.load_pb_model(model_path)
        self.sess = tf.Session(graph=detection_graph)
        self.num_detection = detection_graph.get_tensor_by_name('num_detections:0')
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.masks = detection_graph.get_tensor_by_name('detection_masks:0')

    def detect_object(self, image):
        detection_boxes = tf.squeeze(self.boxes, [0])
        detection_masks = tf.squeeze(self.masks, [0])
        real_num_detection = tf.cast(self.num_detection[0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        self.masks = tf.expand_dims(detection_masks_reframed, 0)

        output_dict = self.sess.run({'num_detections': self.num_detection,
                                     'detection_boxes': self.boxes,
                                     'detection_scores': self.scores,
                                     'detection_classes': self.classes,
                                     'detection_masks': self.masks},
                                    feed_dict={self.image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict
