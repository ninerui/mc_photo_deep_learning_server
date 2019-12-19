# !/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import math
import traceback

import cv2
import numpy as np
from PIL import Image

# P_A: 所有像素，B_B: 外边界， B_S: 内边界
P_A = 1035
B_B = 30
B_S = 15
P_2_1 = (P_A - (B_B * 2) - B_S) // 2
P_2_2 = P_2_1 * 2 + B_S
P_3_1 = (P_A - (B_B * 2) - (B_S * 2)) // 3
P_3_2 = P_3_1 * 2 + B_S
P_3_3 = P_3_1 * 3 + B_S * 2


def resize_image(image, size=(511, 511), face=tuple()):
    scale_w = size[0] / image.size[0]
    scale_h = size[1] / image.size[1]
    if scale_w > scale_h:  # 裁上下
        image = image.resize((size[0], int(scale_w * image.size[1])), Image.ANTIALIAS)
        if len(face) == 0:
            start_pix = (image.size[1] - size[1]) // 2
        else:
            start_pix = max(face[1] - 0.1, 0) * image.size[1]
        image = image.crop((0, start_pix, size[0], size[1] + start_pix))
    else:  # 裁左右
        image = image.resize((int(scale_h * image.size[0]), size[1]), Image.ANTIALIAS)
        if len(face) == 0:
            start_pix = (image.size[0] - size[0]) // 2
        else:
            start_pix = max(face[0] - 0.1, 0) * image.size[0]
            if start_pix + size[0] > image.size[0]:
                start_pix = image.size[0] - size[0]
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


def create_cover_from_2_img(image_list, face_list):
    # image_list = [read_img(i) for i in img_path_list]
    image_direction_list = [(i.size[0] / i.size[1]) - 1. for i in image_list]
    res_img = Image.new(mode='RGB', size=(P_A, P_A), color='white')
    if sum(image_direction_list) > 0:  # 上下结构
        img_1 = image_list.pop(0)
        img_1_face = face_list.pop(0)
        img_1 = resize_image(img_1, size=(P_2_2, P_2_1), face=img_1_face)

        img_2 = image_list.pop(0)
        img_2_face = face_list.pop(0)
        img_2 = resize_image(img_2, size=(P_2_2, P_2_1), face=img_2_face)
        res_img.paste(img_1, box=(B_B, B_B))
        res_img.paste(img_2, box=(B_B, B_B + P_2_1 + B_S))
    else:
        img_1 = image_list.pop(0)
        img_1 = resize_image(img_1, size=(P_2_1, P_2_2))
        img_2 = image_list.pop(0)
        img_2 = resize_image(img_2, size=(P_2_1, P_2_2))
        res_img.paste(img_1, box=(B_B, B_B))
        res_img.paste(img_2, box=(B_B + P_2_1 + B_S, B_B))
    return res_img


def create_cover_from_3_img(image_list, face_list):
    # image_list = [read_img(i) for i in img_path_list]
    image_direction_list = [(i.size[0] - i.size[1]) for i in image_list]

    hx_0_idx = image_direction_list.index(max(image_direction_list))
    image_direction_list.pop(hx_0_idx)
    img_0 = image_list.pop(hx_0_idx)
    img_0_face = face_list.pop(0)
    img_0 = resize_image(img_0, size=(P_2_2, P_2_1), face=img_0_face)

    img_1 = image_list.pop(0)
    img_1_face = face_list.pop(0)
    img_1 = resize_image(img_1, size=(P_2_1, P_2_1), face=img_1_face)

    img_2 = image_list.pop(0)
    img_2_face = face_list.pop(0)
    img_2 = resize_image(img_2, size=(P_2_1, P_2_1), face=img_2_face)

    res_img = Image.new(mode='RGB', size=(P_A, P_A), color='white')
    res_img.paste(img_0, box=(B_B, B_B))
    res_img.paste(img_1, box=(B_B, B_B + P_2_1 + B_S))
    res_img.paste(img_2, box=(B_B + P_2_1 + B_S, B_B + P_2_1 + B_S))
    return res_img


def create_cover_from_4_img(image_list, face_list, out_margin=B_B, in_margin=B_S):
    # image_list = [read_img(i) for i in img_path_list]

    img_1 = image_list.pop(0)
    img_1_face = face_list.pop(0)
    img_1 = resize_image(img_1, size=(P_2_1, P_2_1), face=img_1_face)

    img_2 = image_list.pop(0)
    img_2_face = face_list.pop(0)
    img_2 = resize_image(img_2, size=(P_2_1, P_2_1), face=img_2_face)

    img_3 = image_list.pop(0)
    img_3_face = face_list.pop(0)
    img_3 = resize_image(img_3, size=(P_2_1, P_2_1), face=img_3_face)

    img_4 = image_list.pop(0)
    img_4_face = face_list.pop(0)
    img_4 = resize_image(img_4, size=(P_2_1, P_2_1), face=img_4_face)

    res_img = Image.new(mode='RGB', size=(P_2_1 * 2 + in_margin + out_margin * 2, P_A), color='white')
    res_img.paste(img_1, box=(out_margin, out_margin + P_2_1 + in_margin))
    res_img.paste(img_2, box=(out_margin, out_margin))
    res_img.paste(img_3, box=(out_margin + P_2_1 + in_margin, out_margin + P_2_1 + in_margin))
    res_img.paste(img_4, box=(out_margin + P_2_1 + in_margin, out_margin))
    return res_img


def create_cover_from_5_img(image_list, face_list):
    # image_list = [read_img(i) for i in img_path_list]
    image_direction_list = [(i.size[0] - i.size[1]) for i in image_list]
    hx_0_idx = image_direction_list.index(max(image_direction_list))
    image_direction_list.pop(hx_0_idx)
    img_0 = image_list.pop(hx_0_idx)
    img_0_face = face_list.pop(0)
    img_0 = resize_image(img_0, size=(P_2_2, P_2_1), face=img_0_face)
    img_2_2 = create_cover_from_4_img(image_list, face_list, out_margin=0)
    res_img = Image.new(mode='RGB', size=(P_2_1 * 2 + B_B * 2 + B_S, B_B * 2 + P_2_1 * 3 + B_S * 2), color='white')
    res_img.paste(img_0, box=(B_B, B_B))
    res_img.paste(img_2_2, box=(B_B, B_B + P_2_1 + B_S))
    return res_img


def create_cover_from_6_img(image_list, face_list):
    img_0 = image_list.pop(0)
    img_0_face = face_list.pop(0)
    img_0 = resize_image(img_0, size=(P_3_2, P_3_2), face=img_0_face)
    img_1 = image_list.pop(0)
    img_1_face = face_list.pop(0)
    img_1 = resize_image(img_1, size=(P_3_1, P_3_1), face=img_1_face)
    img_2 = image_list.pop(0)
    img_2_face = face_list.pop(0)
    img_2 = resize_image(img_2, size=(P_3_1, P_3_1), face=img_2_face)
    img_3 = image_list.pop(0)
    img_3_face = face_list.pop(0)
    img_3 = resize_image(img_3, size=(P_3_1, P_3_1), face=img_3_face)
    img_4 = image_list.pop(0)
    img_4_face = face_list.pop(0)
    img_4 = resize_image(img_4, size=(P_3_1, P_3_1), face=img_4_face)
    img_5 = image_list.pop(0)
    img_5_face = face_list.pop(0)
    img_5 = resize_image(img_5, size=(P_3_1, P_3_1), face=img_5_face)
    res_img = Image.new(
        mode='RGB', size=(B_B + P_3_1 + B_S + P_3_1 + B_S + P_3_1 + B_B, B_B + P_3_1 + B_S + P_3_1 + B_S + P_3_1 + B_B),
        color='white')
    res_img.paste(img_0, box=(B_B + P_3_1 + B_S, B_B + P_3_1 + B_S))
    res_img.paste(img_1, box=(B_B, B_B))
    res_img.paste(img_2, box=(B_B, B_B + P_3_1 + B_S))
    res_img.paste(img_3, box=(B_B, B_B + P_3_1 + B_S + P_3_1 + B_S))
    res_img.paste(img_4, box=(B_B + P_3_1 + B_S, B_B))
    res_img.paste(img_5, box=(B_B + P_3_1 + B_S + P_3_1 + B_S, B_B))
    return res_img


def create_cover_from_7_img(image_list, face_list):
    # image_list = [read_img(i) for i in img_path_list]
    image_direction_list = [(i.size[0] - i.size[1]) for i in image_list]

    hx_0_idx = image_direction_list.index(max(image_direction_list))
    image_direction_list.pop(hx_0_idx)
    img_0 = image_list.pop(hx_0_idx)
    img_0_face = face_list.pop(0)
    img_0 = resize_image(img_0, size=(P_3_2, P_3_1), face=img_0_face)

    hx_1_idx = image_direction_list.index(max(image_direction_list))
    image_direction_list.pop(hx_1_idx)
    img_1 = image_list.pop(hx_1_idx)
    img_1_face = face_list.pop(hx_1_idx)
    img_1 = resize_image(img_1, size=(P_3_2, P_3_1), face=img_1_face)

    img_2 = image_list.pop(0)
    img_2_face = face_list.pop(0)
    img_2 = resize_image(img_2, size=(P_3_2, P_3_2), face=img_2_face)

    img_3 = image_list.pop(0)
    img_3_face = face_list.pop(0)
    img_3 = resize_image(img_3, size=(P_3_1, P_3_1), face=img_3_face)

    img_4 = image_list.pop(0)
    img_4_face = face_list.pop(0)
    img_4 = resize_image(img_4, size=(P_3_1, P_3_1), face=img_4_face)

    img_5 = image_list.pop(0)
    img_5_face = face_list.pop(0)
    img_5 = resize_image(img_5, size=(P_3_1, P_3_1), face=img_5_face)

    img_6 = image_list.pop(0)
    img_6_face = face_list.pop(0)
    img_6 = resize_image(img_6, size=(P_3_1, P_3_1), face=img_6_face)

    res_img = Image.new(mode='RGB', size=(B_B * 2 + P_3_1 * 3 + B_S * 2, B_B * 2 + P_3_1 * 4 + B_S * 3), color='white')
    res_img.paste(img_0, box=(B_B, B_B))
    res_img.paste(img_1, box=(B_B + P_3_1 + B_S, B_B + P_3_1 * 3 + B_S * 3))
    res_img.paste(img_2, box=(B_B + P_3_1 + B_S, B_B + P_3_1 + B_S))
    res_img.paste(img_3, box=(B_B + P_3_1 * 2 + B_S * 2, B_B))
    res_img.paste(img_4, box=(B_B, B_B + P_3_1 + B_S))
    res_img.paste(img_5, box=(B_B, B_B + P_3_1 * 2 + B_S * 2))
    res_img.paste(img_6, box=(B_B, B_B + P_3_1 * 3 + B_S * 3))
    return res_img


def create_cover_from_9_img(image_list, face_list):
    img_1 = image_list.pop(0)
    img_1_face = face_list.pop(0)
    img_1 = resize_image(img_1, size=(P_3_1, P_3_1), face=img_1_face)

    img_2 = image_list.pop(0)
    img_2_face = face_list.pop(0)
    img_2 = resize_image(img_2, size=(P_3_1, P_3_1), face=img_2_face)

    img_3 = image_list.pop(0)
    img_3_face = face_list.pop(0)
    img_3 = resize_image(img_3, size=(P_3_1, P_3_1), face=img_3_face)

    img_4 = image_list.pop(0)
    img_4_face = face_list.pop(0)
    img_4 = resize_image(img_4, size=(P_3_1, P_3_1), face=img_4_face)

    img_5 = image_list.pop(0)
    img_5_face = face_list.pop(0)
    img_5 = resize_image(img_5, size=(P_3_1, P_3_1), face=img_5_face)

    img_6 = image_list.pop(0)
    img_6_face = face_list.pop(0)
    img_6 = resize_image(img_6, size=(P_3_1, P_3_1), face=img_6_face)

    img_7 = image_list.pop(0)
    img_7_face = face_list.pop(0)
    img_7 = resize_image(img_7, size=(P_3_1, P_3_1), face=img_7_face)

    img_8 = image_list.pop(0)
    img_8_face = face_list.pop(0)
    img_8 = resize_image(img_8, size=(P_3_1, P_3_1), face=img_8_face)

    img_9 = image_list.pop(0)
    img_9_face = face_list.pop(0)
    img_9 = resize_image(img_9, size=(P_3_1, P_3_1), face=img_9_face)

    res_img = Image.new(mode='RGB', size=(P_A, P_A), color='white')
    res_img.paste(img_1, box=(B_B, B_B))
    res_img.paste(img_2, box=(B_B, B_B + P_3_1 + B_S))
    res_img.paste(img_3, box=(B_B, B_B + P_3_1 + B_S + P_3_1 + B_S))
    res_img.paste(img_4, box=(B_B + P_3_1 + B_S, B_B))
    res_img.paste(img_5, box=(B_B + P_3_1 + B_S + P_3_1 + B_S, B_B))
    res_img.paste(img_6, box=(B_B + P_3_1 + B_S, B_B + P_3_1 + B_S))
    res_img.paste(img_7, box=(B_B + P_3_1 + B_S, B_B + P_3_1 + B_S + P_3_1 + B_S))
    res_img.paste(img_8, box=(B_B + P_3_1 + B_S + P_3_1 + B_S, B_B + P_3_1 + B_S))
    res_img.paste(img_9, box=(B_B + P_3_1 + B_S + P_3_1 + B_S, B_B + P_3_1 + B_S + P_3_1 + B_S))
    return res_img


if __name__ == '__main__':
    try:
        args = sys.argv[1:]
        output_path = args.pop(-1)
        img_path_list = args
        img_count = len(img_path_list)
        assert 2 <= img_count <= 9
        face_cascade = cv2.CascadeClassifier()
        face_cascade.load('./haarcascade_frontalface_default.xml')

        image_list = [read_img(i) for i in img_path_list]

        face_list = []
        for image in image_list:
            img_w, img_h = image.size
            image_rgb = np.array(image)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            image_scale = min(math.sqrt(16000000 / (img_w * img_h)), 1.)
            image_gray = cv2.resize(image_gray, (0, 0), fx=image_scale, fy=image_scale)
            img_w, img_h = image.shape
            faces = face_cascade.detectMultiScale(image_gray)
            if len(faces) == 0:
                face_list.append(tuple())
            else:
                max_face_index = int(np.argmax([i[2] * i[3] for i in faces]))
                max_face_box = faces[max_face_index]
                face_list.append((max_face_box[0] / img_w, max_face_box[1] / img_h,
                                  (max_face_box[0] + max_face_box[2]) / img_w,
                                  (max_face_box[1] + max_face_box[3]) / img_h))  # (left, top, right, bottom)

        if img_count == 2:
            res_img = create_cover_from_2_img(image_list, face_list)
        elif img_count == 3:
            res_img = create_cover_from_3_img(image_list, face_list)
        elif img_count == 4:
            res_img = create_cover_from_4_img(image_list, face_list)
        elif img_count == 5:
            res_img = create_cover_from_5_img(image_list, face_list)
        elif img_count == 6:
            res_img = create_cover_from_6_img(image_list, face_list)
        elif img_count == 7:
            res_img = create_cover_from_7_img(image_list, face_list)
        elif img_count == 9:
            res_img = create_cover_from_9_img(image_list, face_list)
        res_img.save(output_path, "JPEG", quality=100)
        sys.stdout.write('111: {}'.format(output_path))
    except Exception as e:
        sys.stdout.write('000')
        traceback.print_exc()
