import os
import glob
import imghdr
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageSequence
from skimage import transform as trans
from urllib.request import urlretrieve


def files(curr_dir='.', ext='*.exe'):
    """当前目录下的文件"""
    for i in glob.glob(os.path.join(curr_dir, ext)):
        yield i


def remove_files(rootdir, ext):
    """删除rootdir目录下的符合的文件"""
    for i in files(rootdir, ext):
        os.remove(i)


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
    subprocess.check_call(['heif-convert', src_file, result_file])


def IsValidImage(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid


def parser_image(image_path, output_dir, image_get_type=None):
    if image_get_type is None:
        image_get_type = imghdr.what(image_path)
    image_id, image_type = os.path.splitext(os.path.basename(image_path))
    if image_get_type in ['jpeg', 'png', 'bmp']:
        is_valid_image = IsValidImage(image_path)
        if is_valid_image:
            res_code = 1
        else:
            res_code = -3
    elif image_get_type == 'gif':
        try:
            new_image_path = os.path.join(output_dir, '{}.png'.format(image_id))
            ImageSequence.Iterator(Image.open(image_path))[0].save(new_image_path)
            res_code = 1
            os.remove(image_path)
            image_path = new_image_path
        except:
            res_code = -5
    elif image_get_type == 'webp':
        new_img_path = os.path.join(output_dir, "{}.jpg".format(image_id))
        subprocess.run(['dwebp', image_path, '-o', new_img_path])
        if os.path.isfile(new_img_path):
            res_code = 1
            os.remove(image_path)
            image_path = new_img_path
        else:
            res_code = -4
    else:
        if image_type.lower() == '.heic':
            new_img_path = os.path.join(output_dir, "{}.jpg".format(image_id))
            subprocess.run(['heif-convert', image_path, new_img_path])
            if os.path.isfile(new_img_path):  # heic only one image
                res_code = 1
                os.remove(image_path)
                image_path = new_img_path
            elif os.path.isfile(os.path.join(output_dir, "{}-1.jpg".format(image_id))):  # heic have many image
                res_code = 1
                os.remove(image_path)
                os.rename(os.path.join(output_dir, "{}-1.jpg".format(image_id)), new_img_path)
                image_path = new_img_path
                remove_files(output_dir, '{}-*.jpg'.format(image_id))  # remove multi image
            else:
                res_code = -2
        elif image_type.lower() in ['.jpeg', '.png', '.bmp', '.jpg']:
            res_code = 1
        else:
            res_code = -6
    return res_code, image_path


PARSER_IMAGE_CODE = {
    -3: "图片验证失败",
    -1: "图片下载失败",
    -2: "heic转换出错",
    -4: "webp转换失败",
    -5: "gif提取失败",
    -6: "未知的图片类型",
    -7: "解析处理函数出错"
}


def download_and_parser_image(image_url, output_dir):
    image_name = os.path.basename(image_url)
    image_path = os.path.join(output_dir, image_name)
    try:
        urlretrieve(image_url, image_path)
    except:
        return {'code': -1, "info": PARSER_IMAGE_CODE.get(-1, None)}
    res_code = 0
    image_get_type = imghdr.what(image_path)
    try:
        res_code, image_path = parser_image(image_path, output_dir, image_get_type)
    except:
        res_code = -7
    finally:
        return {
            'code': res_code,
            "image_path": image_path,
            "img_type": image_get_type,
            "info": PARSER_IMAGE_CODE.get(res_code, None)
        }

    # image_get_type = imghdr.what(image_path)
    # image_id, image_type = os.path.splitext(image_name)
    # if image_get_type == 'webp':
    #     new_img_path = os.path.join(output_dir, "{}.jpg".format(image_id))
    #     subprocess.run(['dwebp', image_path, '-o', new_img_path])
    #     if os.path.isfile(new_img_path):
    #         util.removefile(image_path)
    #         return {'code': 1, "image_path": new_img_path}
    # elif image_get_type in ['jpeg', 'png', 'bmp']:
    #     if IsValidImage(image_path):
    #         return {'code': 1, "image_path": image_path}
    #     else:
    #         return {'code': -3, "image_path": image_path}
    # elif image_get_type == 'gif':
    #     try:
    #         new_image_path = os.path.join(output_dir, '{}.png'.format(image_id))
    #         ImageSequence.Iterator(Image.open(image_path))[0].save(new_image_path)
    #         shutil.rmtree(image_path)
    #         return {'code': 1, "image_path": new_image_path}
    #     except:
    #         return {'code': -4, "image_path": image_path}
    # elif image_get_type is None:
    #     if image_type.lower() == '.heic':
    #         new_img_path = os.path.join(output_dir, "{}.jpg".format(image_id))
    #         subprocess.run(['heif-convert', image_path, new_img_path])
    #         if os.path.isfile(new_img_path):  # heic only one image
    #             res_code = 1
    #             os.remove(image_path)
    #             image_path = new_img_path
    #         elif os.path.isfile(os.path.join(output_dir, "{}-1.jpg".format(image_id))):  # heic have many image
    #             res_code = 1
    #             os.remove(image_path)
    #             os.rename(os.path.join(output_dir, "{}-1.jpg".format(image_id)), new_img_path)
    #             image_path = new_img_path
    #             remove_files(output_dir, '{}-*.jpg'.format(image_id))
    #         else:
    #             res_code = -1
    #         #
    #         #
    #         # tmp_dir = os.path.join(output_dir, image_id)
    #         # util.makedirs(tmp_dir)
    #         # new_img_path = os.path.join(tmp_dir, "{}.jpg".format(image_id))
    #         # subprocess.run(['heif-convert', image_path, new_img_path])
    #         # if os.path.isfile(new_img_path):
    #         #     shutil.move(new_img_path, image_path)
    #         # elif os.path.isfile(os.path.join(tmp_dir, "{}-1.jpg".format(image_id))):
    #         #     shutil.move(os.path.join(tmp_dir, "{}-1.jpg".format(image_id)), image_path)
    #         # shutil.rmtree(tmp_dir)
    #         # shutil.rmtree(image_path)
    #         # return {'code': 1, "image_path": image_path}
    #     elif image_type.lower() in ['.jpeg', '.png', '.bmp', '.jpg']:
    #         res_code = 1
    #
    #         # return {'code': 1, "image_path": image_path}
    # return {'code': -2, "image_path": image_path, "img_type": image_get_type}
