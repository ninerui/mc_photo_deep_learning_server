from flask import Flask, request, send_from_directory
import cv2
import urllib.request
import tensorflow as tf
import threading
import os
import numpy as np
import time
import json

from utils import util
from dl_module import image_style_transform_interface

app = Flask(__name__)


def test_style(image, style):
    image_shape = image.shape
    batch_shape = (1,) + image_shape
    with tf.Graph().as_default(), tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = image_style_transform_interface.net(img_placeholder)
        saver = tf.train.Saver()
        if style == 1:
            saver.restore(sess, "./models/image_style_transfer_models/la_muse.ckpt")
        elif style == 2:
            saver.restore(sess, "./models/image_style_transfer_models/rain_princess.ckpt")
        elif style == 3:
            saver.restore(sess, "./models/image_style_transfer_models/scream.ckpt")
        elif style == 4:
            saver.restore(sess, "./modes/image_style_transfer_models/udnie.ckpt")
        elif style == 5:
            saver.restore(sess, "./models/image_style_transfer_models/wave.ckpt")
        elif style == 6:
            saver.restore(sess, "./models/image_style_transfer_models/wreck.ckpt")
        else:
            return None
        _preds = sess.run(preds, feed_dict={img_placeholder: [image]})
    return _preds[0]


@app.route("/image_style_transfer", methods=['GET', 'POST'])
def image_style_transfer():
    if request.method == 'GET':
        style = request.args.get("style")
        if not style:
            return "没有指定风格"
        style = int(style)
        assert style in [1, 2, 3, 4, 5, 6]
        image_url = request.args.get("image_url")
        if not image_url:
            return "没有图片地址"
        try:
            a = time.time()
            print(image_url)
            resp = urllib.request.urlopen(image_url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            m = max(image.shape[0], image.shape[1])
            f = 1024.0 / m
            if f < 1.0:
                image = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))

            print('xiazaitupian', time.time() - a)

            a = time.time()
            img = test_style(image, style=style)
            all_time = time.time() - a
            print(all_time)
            img = np.clip(img, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(image_url)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return json.dumps({
                "download_url": os.path.join("http://{}/download".format(local_ip), os.path.basename(image_url))
            })
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            return "error"

    else:
        return "还没开放post请求"


def del_image(image_path):
    if os.path.isfile(image_path):
        time.sleep(60)
        os.remove(image_path)
    return


@app.route("/download/<path:filename>")
def downloader(filename):
    threading.Thread(target=del_image, args=(os.path.join(output_dir, filename),)).start()
    return send_from_directory(output_dir, filename, as_attachment=True)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    local_ip = util.get_local_ip()
    output_dir = '../image_style_transfer_output'
    util.makedirs(output_dir)
    # output_dir = r'D:\code_i4\fast-style-transfer\examples'
    app.run(host="127.0.0.1", port="8082")
