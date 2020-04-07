# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import glob
# import imghdr
import logging
import datetime
import subprocess

import cv2
import pyheif
import piexif
import numpy as np
from PIL import Image, ImageSequence, ImageFont, ImageDraw
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
    heif_file = pyheif.read_heif(src_file)
    pi = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
    pi.save(result_file, format='jpeg')


def parser_image(image_path, output_dir):
    if os.path.getsize(image_path) == 0:  # 首先获取文件大小, if size == 0b, 返回代码3
        res_code = 3
        os.remove(image_path)
        return res_code, image_path, None
    import magic
    file_type = magic.from_file(image_path, mime=True)  # 'image/heif', 'image/png', 'image/jpeg', 'image/gif'
    b_class, s_class = file_type.split('/')
    image_id, image_type = os.path.splitext(os.path.basename(image_path))
    new_dir = os.path.join(output_dir, image_id)
    if not os.path.exists(new_dir): os.makedirs(new_dir)
    new_img_path = os.path.join(new_dir, "{}.jpg".format(image_id))
    if b_class == 'image':
        if s_class == 'webp':
            new_img_path = os.path.join(output_dir, image_id, "{}.jpg".format(image_id))
            try:
                subprocess.run(['dwebp', image_path, '-o', new_img_path])
            except Exception as e:
                logging.exception(e)
            finally:
                res_code = 1 if os.path.exists(new_img_path) else -4
        elif s_class in ['heic', 'heif']:
            try:
                heic2jpg(image_path, new_img_path)
            except Exception as e:
                logging.exception(e)
            finally:
                res_code = 1 if os.path.exists(new_img_path) else -2
        elif s_class in ['jpeg', 'png', 'bmp']:
            try:
                Image.open(image_path).load()
                shutil.copyfile(image_path, new_img_path)
            except OSError:
                cv2.imwrite(new_img_path, cv2.imread(image_path))
            except Exception as e:
                logging.exception(e)
            finally:
                res_code = 1 if os.path.exists(new_img_path) else -3
        # elif s_class == 'gif':
        #     try:
        #         ImageSequence.Iterator(Image.open(image_path))[0].save(new_img_path)
        #     except Exception as e:
        #         logging.exception(e)
        #     finally:
        #         res_code = 1 if os.path.exists(new_img_path) else -5
        else:
            res_code = -9
    else:
        res_code = -8
    if os.path.exists(image_path) and res_code == 1:
        os.remove(image_path)

    # image_get_type = imghdr.what(image_path)
    # image_id, image_type = os.path.splitext(os.path.basename(image_path))
    # if image_get_type in ['jpeg', 'png', 'bmp']:
    #     try:
    #         Image.open(image_path).load()
    #         res_code = 1
    #     except OSError:
    #         try:
    #             cv2.imwrite(image_path, cv2.imread(image_path))
    #             res_code = 1
    #         except cv2.error:
    #             res_code = -3
    # elif image_get_type == 'gif':
    #     try:
    #         new_image_path = os.path.join(output_dir, '{}.png'.format(image_id))
    #         ImageSequence.Iterator(Image.open(image_path))[0].save(new_image_path)
    #         res_code = 1
    #         os.remove(image_path)
    #         image_path = new_image_path
    #     except Exception as e:
    #         logging.exception(e)
    #         res_code = -5
    # elif image_get_type == 'webp':
    #     new_img_path = os.path.join(output_dir, "{}.jpg".format(image_id))
    #     subprocess.run(['dwebp', image_path, '-o', new_img_path])
    #     if os.path.isfile(new_img_path):
    #         res_code = 1
    #         if image_path != new_img_path:
    #             os.remove(image_path)
    #         # os.remove(image_path)
    #         image_path = new_img_path
    #     else:
    #         res_code = -4
    # else:
    # if image_type.lower() == '.heic':
    #     new_img_path = os.path.join(output_dir, "{}.jpg".format(image_id))
    #     try:
    #         heic2jpg(image_path, new_img_path)
    #     except Exception as e:
    #         logging.exception(e)
    #     finally:
    #         if os.path.isfile(new_img_path):
    #             res_code = 1
    #             os.remove(image_path)
    #             # if image_path != new_img_path:
    #             #     os.remove(image_path)
    #             image_path = new_img_path
    #         else:
    #             res_code = -2
    # elif image_type.lower() in ['.jpeg', '.png', '.bmp', '.jpg']:
    #     res_code = -8
    # # elif image_type.lower() in ['.mp4']:
    # #     res_code = 2
    # else:
    #     res_code = -6
    return res_code, new_img_path, file_type


PARSER_IMAGE_CODE = {
    -3: "image valid fail",
    -1: "image download fail",
    -2: "heic convert fail",
    -4: "webp convert fail",
    -5: "gif get image fail",
    -6: "unknown image type",
    -7: "function error",
    -8: "imghdr check None, but is .jpg",
    3: "image is zero size"
}


def download_and_parser_image(image_url, output_dir):
    image_name = os.path.basename(image_url)
    image_path = os.path.join(output_dir, image_name)
    try:
        urlretrieve(image_url, image_path)
    except Exception as e:
        logging.exception(e)
        return {'code': -1, "info": PARSER_IMAGE_CODE.get(-1, None)}
    res_code = 0
    image_get_type = None
    try:
        res_code, image_path, image_get_type = parser_image(image_path, output_dir)
    except Exception as e:
        logging.exception(e)
        res_code = -7
    finally:
        return {
            'code': res_code,
            "image_path": image_path,
            "img_type": image_get_type,
            "info": PARSER_IMAGE_CODE.get(res_code, None)
        }


def resize_image(image, size=(511, 511)):
    scale_w = size[0] / image.size[0]
    scale_h = size[1] / image.size[1]
    if scale_w > scale_h:
        image = image.resize((size[0], int(scale_w * image.size[1])))
        start_pix = (image.size[1] - size[1]) // 2
        image = image.crop((0, start_pix, size[0], size[1] + start_pix))
    else:
        image = image.resize((int(scale_h * image.size[0]), size[1]))
        start_pix = (image.size[0] - size[0]) // 2
        image = image.crop((start_pix, 0, size[0] + start_pix, size[1]))
    return image


def read_img(img_path):
    img = Image.open(img_path)
    exif = dict(img.getexif().items())
    orientation = exif.get(274, None)  # 图片方向信息存在274里面
    if orientation == 3:
        img = img.rotate(180, expand=True)
    elif orientation == 6:
        img = img.rotate(270, expand=True)
    elif orientation == 8:
        img = img.rotate(90, expand=True)
    return img


def create_past_now_img(img_path_list, img_time_list, human_list, output_path):
    image_list = [read_img(i) for i in img_path_list]
    image_direction_list = [(i.size[0] / i.size[1]) - 1. for i in image_list]
    if sum(image_direction_list) > 0:
        img_1 = image_list.pop(0)
        resize_h1 = int(img_1.size[1] * (690 / img_1.size[0]))
        img_1 = img_1.resize((690, resize_h1))
        human_1 = human_list.pop(0)

        img_2 = image_list.pop(0)
        resize_h2 = int(img_2.size[1] * (690 / img_2.size[0]))
        img_2 = img_2.resize((690, resize_h2))
        human_2 = human_list.pop(0)

        resize_h = min(resize_h1, resize_h2)
        # human_h1 = (int(str(human_0)[2]) - int(str(human_0)[0]) / 10) * resize_h1
        try:
            human_mid1 = ((int(str(human_1)[2]) - int(str(human_1)[0])) / 10) * resize_h1
        except:
            human_mid1 = resize_h1 // 2
        human_ymin1 = max((human_mid1 - resize_h // 2), 0)
        human_ymax1 = min((human_mid1 + resize_h // 2), resize_h1)
        img_1 = img_1.crop((0, human_ymin1, 690, human_ymax1)).resize((690, resize_h)).convert('RGBA')
        try:
            human_mid2 = ((int(str(human_2)[2]) - int(str(human_2)[0])) / 10) * resize_h2
        except:
            human_mid2 = resize_h2 // 2
        human_ymin2 = max((human_mid2 - resize_h // 2), 0)
        human_ymax2 = min((human_mid2 + resize_h // 2), resize_h2)
        img_2 = img_2.crop((0, human_ymin2, 690, human_ymax2)).resize((690, resize_h)).convert('RGBA')

        txt_mask = Image.new('RGBA', img_1.size, (0, 0, 0, 0))
        g_a = (255 * 0.52) / int(txt_mask.size[1] * 0.45)
        lala = 0
        for i in range(int(txt_mask.size[1] * 0.55), txt_mask.size[1]):
            for j in range(txt_mask.size[0]):
                txt_mask.putpixel((j, i), (0, 0, 0, int(lala * g_a)))
            lala += 1
        img_1 = Image.alpha_composite(img_1, txt_mask)
        img_1 = Image.alpha_composite(img_1, create_text_mask_row_linshi(img_1.size, img_time_list.pop(0)))

        img_2 = Image.alpha_composite(img_2, txt_mask)
        img_2 = Image.alpha_composite(img_2, create_text_mask_row_linshi(img_2.size, img_time_list.pop(0)))

        res_img = Image.new(mode='RGB', size=(690, resize_h + resize_h + 10), color='white')
        res_img.paste(img_1, box=(20, 20))
        res_img.paste(img_2, box=(20, resize_h + 10 + 20))
    else:
        img_1 = image_list.pop(0)
        resize_h1 = int(img_1.size[1] * (320 / img_1.size[0]))
        img_1 = img_1.resize((320, resize_h1))
        human_1 = human_list.pop(0)

        img_2 = image_list.pop(0)
        resize_h2 = int(img_2.size[1] * (320 / img_2.size[0]))
        img_2 = img_2.resize((320, resize_h2))
        human_2 = human_list.pop(0)

        resize_h = min(resize_h1, resize_h2)
        try:
            human_mid1 = ((int(str(human_1)[2]) - int(str(human_1)[0])) / 10) * resize_h1
        except:
            human_mid1 = resize_h1 // 2
        human_ymin1 = max((human_mid1 - resize_h // 2), 0)
        human_ymax1 = min((human_mid1 + resize_h // 2), resize_h1)
        img_1 = img_1.crop((0, human_ymin1, 320, human_ymax1)).resize((320, resize_h)).convert('RGBA')
        try:
            human_mid2 = ((int(str(human_2)[2]) - int(str(human_2)[0])) / 10) * resize_h2
        except:
            human_mid2 = resize_h2 // 2
        human_ymin2 = max((human_mid2 - resize_h // 2), 0)
        human_ymax2 = min((human_mid2 + resize_h // 2), resize_h2)
        img_2 = img_2.crop((0, human_ymin2, 320, human_ymax2)).resize((320, resize_h)).convert('RGBA')

        txt_mask = Image.new('RGBA', img_1.size, (0, 0, 0, 0))
        g_a = (255 * 0.52) / int(txt_mask.size[1] * 0.45)
        lala = 0
        for i in range(int(txt_mask.size[1] * 0.55), txt_mask.size[1]):
            for j in range(txt_mask.size[0]):
                txt_mask.putpixel((j, i), (0, 0, 0, int(lala * g_a)))
            lala += 1

        img_1 = Image.alpha_composite(img_1, txt_mask)
        img_1 = Image.alpha_composite(img_1, create_text_mask_linshi(img_1.size, img_time_list.pop(0)))

        img_2 = Image.alpha_composite(img_2, txt_mask)
        img_2 = Image.alpha_composite(img_2, create_text_mask_linshi(img_2.size, img_time_list.pop(0)))

        res_img = Image.new(mode='RGB', size=(690, resize_h + 40), color='white')
        res_img.paste(img_1, box=(20, 20))
        res_img.paste(img_2, box=(320 + 10 + 20, 20))
    res_img.save(output_path, format='png', quality=95)


def create_text_mask_linshi(img_size, timestamp):
    photo_time = datetime.datetime.fromtimestamp(timestamp)
    txt_mask = Image.new('RGBA', img_size, (255, 255, 255, 0))
    unicode_font_20 = ImageFont.truetype(font='data/PingFang-SC-Bold.ttf', size=16)
    unicode_font_30 = ImageFont.truetype(font='data/PingFang-SC-Bold.ttf', size=28)
    draw = ImageDraw.Draw(txt_mask)
    text00 = u"{}".format(photo_time.year)
    text01 = r'年'
    text_size_00 = unicode_font_30.getsize(text00)
    text_size_01 = unicode_font_20.getsize(text01)
    text_size_0 = (text_size_00[0] + text_size_01[0], text_size_00[1])
    # text_coordinate_00 = int((img_size[0] - text_size_0[0]) / 2), int((img_size[1] * 0.96) - text_size_00[1])
    text_coordinate_00 = int((img_size[0] - text_size_0[0]) / 2), img_size[1] - 99
    draw.text(text_coordinate_00, text00, font=unicode_font_30, fill=(255, 255, 255, 200))
    # text_coordinate_01 = int((img_size[0] - text_size_0[0]) / 2) + text_size_00[0], int((img_size[1] * 0.96) - text_size_01[1])
    text_coordinate_01 = int((img_size[0] - text_size_0[0]) / 2) + text_size_00[0], img_size[1] - 99 + (
            text_size_00[1] - text_size_01[1])
    draw.text(text_coordinate_01, text01, font=unicode_font_20, fill=(255, 255, 255, 200))
    text1 = u"· {}月{}号 ·".format(photo_time.month, photo_time.day)
    text_width = unicode_font_20.getsize(text1)
    # text_coordinate = int((img_size[0] - text_width[0]) / 2), int((img_size[1] * 0.9882) - text_width[1])
    text_coordinate = int((img_size[0] - text_width[0]) / 2), img_size[1] - 48
    draw.text(text_coordinate, text1, font=unicode_font_20, fill=(255, 255, 255, 170))
    return txt_mask


def create_text_mask_row_linshi(img_size, timestamp):
    photo_time = datetime.datetime.fromtimestamp(timestamp)
    txt_mask = Image.new('RGBA', img_size, (255, 255, 255, 0))
    unicode_font_20 = ImageFont.truetype(font='data/PingFang-SC-Bold.ttf', size=16)
    unicode_font_30 = ImageFont.truetype(font='data/PingFang-SC-Bold.ttf', size=28)
    draw = ImageDraw.Draw(txt_mask)
    text00 = u"{}".format(photo_time.year)
    text01 = r'年'
    text_size_00 = unicode_font_30.getsize(text00)
    text_size_01 = unicode_font_20.getsize(text01)
    text_size_0 = (text_size_00[0] + text_size_01[0], text_size_00[1])
    # text_coordinate_00 = int(img_size[0] * 0.02), int((511 * 0.97) - text_size_00[1])
    text_coordinate_00 = 26, int(img_size[1] * 0.84)
    draw.text(text_coordinate_00, text00, font=unicode_font_30, fill=(255, 255, 255, 200))

    # text_coordinate_01 = int((img_size[0] * 0.02) + text_size_00[0]), int((511 * 0.97) - text_size_01[1])
    text_coordinate_01 = int(26 + text_size_00[0]), int(img_size[1] * 0.84 + (text_size_00[1] - text_size_01[1]))
    draw.text(text_coordinate_01, text01, font=unicode_font_20, fill=(255, 255, 255, 200))

    text1 = u"· {}月{}号".format(photo_time.month, photo_time.day)
    text_width = unicode_font_20.getsize(text1)

    # text_coordinate = int((img_size[0] * 0.98) - text_width[0]), int((511 * 0.97) - text_width[1])
    text_coordinate = 539, int(img_size[1] * 0.87)
    draw.text(text_coordinate, text1, font=unicode_font_20, fill=(255, 255, 255, 170))
    return txt_mask


def push_data_to_dict(source_dict, output_dict, key):
    data = source_dict.get(key, None)
    if not data:
        return output_dict
    output_dict[key] = data
    return output_dict


def save_image_with_exif(source_img):
    new_exif_dict = {}
    exif_data = source_img.info.get("exif", None)
    if not exif_data:
        return None
    exif_dict = piexif.load(exif_data)
    source_0th = exif_dict.get("0th", None)
    if source_0th:
        dict_0th = {}
        dict_0th = push_data_to_dict(source_0th, dict_0th, piexif.ImageIFD.Model)
        dict_0th = push_data_to_dict(source_0th, dict_0th, piexif.ImageIFD.Make)
        dict_0th = push_data_to_dict(source_0th, dict_0th, piexif.ImageIFD.DateTime)
        # dict_0th = push_data_to_dict(source_0th, dict_0th, piexif.ImageIFD.Orientation)
        if len(dict_0th) != 0:
            new_exif_dict["0th"] = dict_0th
    source_exif = exif_dict.get("Exif", None)
    if source_exif:
        dict_exif = {}
        dict_exif = push_data_to_dict(source_exif, dict_exif, piexif.ExifIFD.ExifVersion)
        dict_exif = push_data_to_dict(source_exif, dict_exif, piexif.ExifIFD.DateTimeOriginal)
        dict_exif = push_data_to_dict(source_exif, dict_exif, piexif.ExifIFD.ColorSpace)
        if len(dict_exif) != 0:
            new_exif_dict["Exif"] = dict_exif
    source_gps = exif_dict.get("GPS", None)
    if source_gps:
        new_exif_dict["GPS"] = source_gps
    source_interop = exif_dict.get("Interop", None)
    if source_interop:
        new_exif_dict["Interop"] = source_interop
    source_thumbnail = exif_dict.get("thumbnail", None)
    if source_thumbnail:
        new_exif_dict["thumbnail"] = source_thumbnail
    if len(new_exif_dict) != 0:
        return piexif.dump(new_exif_dict)
    else:
        return None
