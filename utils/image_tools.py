import os
import imghdr
import shutil
import subprocess

import cv2
# import pyheif
import numpy as np
from PIL import Image
from skimage import transform as trans
from urllib.request import urlretrieve

from utils import util


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def preprocess(img, image_size, bbox=None, landmark=None, **kwargs):
    M = None
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped


def heic2jpg(src_file, result_file):
    # # 脚本命令行转
    # params = ['convert', src_file, result_file]
    subprocess.check_call(['heif-convert', src_file, result_file])

    # python读取后转
    # heif_file = pyheif.read_heif(src_file)
    # pi = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
    # pi.save(result_file, format='jpeg')
    # img = np.fromstring(heif_file.data, dtype=np.uint8).reshape((heif_file.size[1], heif_file.size[0], 3))
    # cv2.imwrite(result_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def IsValidImage(file):
    bValid = True
    if isinstance(file, (str, os.PathLike)):
        fileObj = open(file, 'rb')
    else:
        fileObj = file
    buf = fileObj.read()
    if buf[6:10] in (b'JFIF', b'Exif'):  # jpg图片
        if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
            bValid = False
    else:
        try:
            Image.open(fileObj).verify()
        except:
            bValid = False
    return bValid


def download_image(image_url, output_dir):
    image_name = os.path.basename(image_url)
    image_path = os.path.join(output_dir, image_name)
    try:
        urlretrieve(image_url, image_path)
    except:
        return {'code': -1}
    image_get_type = imghdr.what(image_path)
    image_id, image_type = os.path.splitext(image_name)
    if image_get_type == 'webp':
        new_img_path = os.path.join(output_dir, "{}.jpg".format(image_id))
        subprocess.run(['dwebp', image_path, '-o', new_img_path])
        if os.path.isfile(new_img_path):
            util.removefile(image_path)
            return {'code': 1, "image_path": new_img_path}
    elif image_get_type in ['jpeg', 'png', 'bmp']:
        if IsValidImage(image_path):
            return {'code': 1, "image_path": image_path}
        else:
            return {'code': -3, "image_path": image_path}
    elif image_get_type is None:
        if image_type.lower() == '.heic':
            tmp_dir = os.path.join(output_dir, image_id)
            util.makedirs(tmp_dir)
            new_img_path = os.path.join(tmp_dir, "{}.jpg".format(image_id))
            subprocess.run(['heif-convert', image_path, new_img_path])
            if os.path.isfile(new_img_path):
                shutil.move(new_img_path, image_path)
                # util.removefile(image_path)
            elif os.path.isfile(os.path.join(tmp_dir, "{}-1.jpg".format(image_id))):
                shutil.move(os.path.join(tmp_dir, "{}-1.jpg".format(image_id)), image_path)
                # shutil.rmtree(os.path.join(output_dir, image_id))
                # return {'code': 1, "image_path": os.path.join(output_dir, image_id, "{}-1.jpg".format(image_id))}
            shutil.rmtree(tmp_dir)
            return {'code': 1, "image_path": image_path}
        elif image_type.lower() in ['.jpeg', '.png', '.bmp', '.jpg']:
            return {'code': 1, "image_path": image_path}
    return {'code': -2, "image_path": image_path, "img_type": image_get_type}
