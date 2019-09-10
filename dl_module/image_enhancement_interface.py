import os

import cv2
import numpy as np
import skimage
import skimage.io
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn

from utils import tf_tools


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def unet(input):
    with tf.variable_scope("generator"):
        input1 = slim.conv2d(input, 16, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv')
        conv1 = slim.conv2d(input1, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
        pool1 = slim.conv2d(conv1, 16, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1')

        conv2 = slim.conv2d(pool1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
        pool2 = slim.conv2d(conv2, 32, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='pooling2')

        conv3 = slim.conv2d(pool2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
        pool3 = slim.conv2d(conv3, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='pooling3')

        conv4 = slim.conv2d(pool3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
        pool4 = slim.conv2d(conv4, 128, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='pooling4')

        conv5 = slim.conv2d(pool4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
        conv_global = tf.reduce_mean(conv5, axis=[1, 2])
        conv_dense = tf.layers.dense(conv_global, units=128, activation=tf.nn.relu)
        feature = tf.expand_dims(conv_dense, axis=1)
        feature = tf.expand_dims(feature, axis=2)
        ones = tf.zeros(shape=tf.shape(conv4))
        global_feature = feature + ones

        up6 = tf.concat([conv4, global_feature], axis=3)
        conv6 = slim.conv2d(up6, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

        up7 = upsample_and_concat(conv6, conv3, 64, 128)
        conv7 = slim.conv2d(up7, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

        up8 = upsample_and_concat(conv7, conv2, 32, 64)
        conv8 = slim.conv2d(up8, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

        up9 = upsample_and_concat(conv8, conv1, 16, 32)
        conv9 = slim.conv2d(up9, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

        conv9 = input1 * conv9
        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 16], stddev=0.02))
        conv10 = tf.nn.conv2d_transpose(conv9, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        out = slim.conv2d(conv10, 3, [3, 3], rate=1, activation_fn=nn.tanh, scope='out') * 0.58 + 0.52

    return out


class AIChallengeWithDPEDSRCNN:
    def __init__(self, model_path='./models/image_enhancement_models/ai_challenge_dped_srcnn.pb'):
        graph = tf_tools.load_pb_model(model_path)
        self.sess = tf.Session(graph=graph)
        self.image_tensor = graph.get_tensor_by_name('input:0')
        self.output_tensor = graph.get_tensor_by_name('output:0')

    def get_image(self, images):
        return self.sess.run(self.output_tensor, feed_dict={self.image_tensor: images})


def get_enjancement_img(input_img, output_img):
    im_input = cv2.imread(input_img, -1)
    width, heigh, _ = im_input.shape
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    #     checkpoint_path = tf.train.latest_checkpoint('./models/image_enhancement_models/mt_phone')
    tf.reset_default_graph()
    t_fullres_input = tf.placeholder(tf.float32, (1, width, heigh, 3))
    with tf.variable_scope('inference'):
        prediction = unet(t_fullres_input)
    output = tf.cast(255.0 * tf.squeeze(tf.clip_by_value(prediction, 0, 1)), tf.uint8)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, './models/image_enhancement_models/mt_phone/model.ckpt-482341')
        im_input = np.flip(im_input, 2)
        im_input = skimage.img_as_float(im_input)
        im_input = im_input[np.newaxis, :, :, :]
        feed_dict = {
            t_fullres_input: im_input
        }
        out_ = sess.run(output, feed_dict=feed_dict)
        skimage.io.imsave(output_img, out_)


def get_normalize_size_shape_method(img, max_length):
    [height, width, channels] = img.shape
    if height >= width:
        longerSize = height
        shorterSize = width
    else:
        longerSize = width
        shorterSize = height

    scale = float(max_length) / float(longerSize)

    outputHeight = int(round(height * scale))
    outputWidth = int(round(width * scale))
    return outputHeight, outputWidth


def cpu_normalize_image(img, max_length):
    outputHeight, outputWidth = get_normalize_size_shape_method(img, max_length)
    outputImg = Image.fromarray(img)
    outputImg = outputImg.resize((outputWidth, outputHeight), Image.ANTIALIAS)
    outputImg = np.array(outputImg)
    return outputImg


def normalizeImage(img, max_length):
    [height, width, channels] = img.shape
    max_l = max(height, width)

    is_need_resize = max_l != 512
    if is_need_resize:
        img = cpu_normalize_image(img, max_length)
    return img


def random_pad_to_size(img, size, mask, pad_symmetric, use_random):
    if mask is None:
        mask = np.ones(shape=img.shape)
    s0 = size - img.shape[0]
    s1 = size - img.shape[1]

    if use_random:
        b0 = np.random.randint(0, s0 + 1)
        b1 = np.random.randint(0, s1 + 1)
    else:
        b0 = 0
        b1 = 0
    a0 = s0 - b0
    a1 = s1 - b1
    if pad_symmetric:
        img = np.pad(img, ((b0, a0), (b1, a1), (0, 0)), 'symmetric')
    else:
        img = np.pad(img, ((b0, a0), (b1, a1), (0, 0)), 'constant')
    mask = np.pad(mask, ((b0, a0), (b1, a1), (0, 0)), 'constant')
    return img, mask, [b0, img.shape[0] - a0, b1, img.shape[1] - a1]


def safe_casting(data, dtype):
    output = np.clip(data + 0.5, np.iinfo(dtype).min, np.iinfo(dtype).max)
    output = output.astype(dtype)
    return output


class ImageHDRs:
    def __init__(self, model_path='./models/image_enhancement_models/model_999_new.pb'):
        graph_def = tf_tools.load_pb_model(model_path)
        self.sess = tf.Session(graph=graph_def)
        self.netG_test_output1 = graph_def.get_tensor_by_name(
            'netG-999/netG-999_var_scope/netG-999_var_scopeA/netG-999_3/Add_48:0')
        self.netG_test_gfeature1 = graph_def.get_tensor_by_name(
            'netG-999/netG-999_var_scope/netG-999_var_scopeA/netG-999_2/BiasAdd_3:0')
        self.input1 = graph_def.get_tensor_by_name('Placeholder:0')
        self.rate1 = graph_def.get_tensor_by_name('Placeholder_2:0')
        self.input2 = graph_def.get_tensor_by_name('Placeholder_1:0')

    def get_hdr_image(self, input_img, output_img):
        input_img = cv2.imread(input_img, 1)
        h, w, _ = input_img.shape
        os.remove(input_img)
        if max(h, w) > 2048:
            if h >= w:
                longerSize = h
            else:
                longerSize = w
            scale = float(2048) / float(longerSize)
            outputHeight = int(round(h * scale))
            outputWidth = int(round(w * scale))
            outputImg = Image.fromarray(input_img)
            outputImg = outputImg.resize((outputWidth, outputHeight), Image.ANTIALIAS)
            outputImg = np.array(outputImg)
            cv2.imwrite(input_img, outputImg)
        else:
            cv2.imwrite(input_img, input_img)
        input_img = cv2.imread(input_img, -1)
        resize_input_img = normalizeImage(input_img, 512)
        resize_input_img, _, _ = random_pad_to_size(resize_input_img, 512, None, True, False)
        resize_input_img = resize_input_img[None, :, :, :]

        gfeature = self.sess.run(
            self.netG_test_gfeature1,
            feed_dict={self.input1: resize_input_img, self.rate1: 1})

        h, w, c = input_img.shape
        rate = int(round(max(h, w) / 512))
        if rate == 0:
            rate = 1
        padrf = rate * 64
        patch = 16 * 64
        pad_h = 0 if h % patch == 0 else patch - (h % patch)
        pad_w = 0 if w % patch == 0 else patch - (w % patch)
        pad_h = pad_h + padrf if pad_h < padrf else pad_h
        pad_w = pad_w + padrf if pad_w < padrf else pad_w

        input_img = np.pad(input_img, [(padrf, pad_h), (padrf, pad_w), (0, 0)], 'reflect')
        y_list = []
        for y in range(padrf, h + padrf, patch):
            x_list = []
            for x in range(padrf, w + padrf, patch):
                crop_img = input_img[None, y - padrf:y + padrf + patch, x - padrf:x + padrf + patch, :]
                enhance_test_img = self.sess.run(
                    self.netG_test_output1,
                    feed_dict={self.input1: crop_img, self.input2: gfeature, self.rate1: rate})
                enhance_test_img = enhance_test_img[0, padrf:-padrf, padrf:-padrf, :]
                x_list.append(enhance_test_img)
            y_list.append(np.concatenate(x_list, axis=1))
        enhance_test_img = np.concatenate(y_list, axis=0)
        enhance_test_img = enhance_test_img[:h, :w, :]
        enhance_test_img = safe_casting(enhance_test_img * tf.as_dtype(np.uint8).max, np.uint8)
        cv2.imwrite(output_img, enhance_test_img)
