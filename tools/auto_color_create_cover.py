#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys

import cv2
import pyheif
import numpy as np
from PIL import Image


def read_img(img_path):
    image_id, image_type = os.path.splitext(img_path)
    if image_type.lower() == '.heic':
        heif_file = pyheif.read_heif(img_path)
        pi = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
    else:
        pi = Image.open(img_path)
    return pi


def parser_2_h(img_path_list, size=(1005, 1005), fill_size=3):
    img_list = [
        read_img(i).resize(((size[0] - fill_size) // 2, size[1])) if type(i) == str else i for i in img_path_list]
    res_w, res_h = img_list[0].size
    res_img = Image.new(mode='RGB', size=size, color=0)
    res_img.paste(img_list[0], box=(0, 0))
    res_img.paste(img_list[1], box=(res_w + fill_size, 0))
    return res_img


def parser_2_v(img_path_list, size=(1005, 1005), fill_size=3):
    img_list = [
        read_img(i).resize((size[1], (size[1] - fill_size) // 2)) if type(i) == str else i for i in img_path_list]
    res_w, res_h = img_list[0].size
    res_img = Image.new(mode='RGB', size=size, color=0)
    res_img.paste(img_list[0], box=(0, 0))
    res_img.paste(img_list[1], box=(0, res_h + fill_size))
    return res_img


def parser_3_v(img_path_list, size=(1005, 1005), fill_size=3):
    img_list = [
        read_img(i).resize((size[1], (size[1] - fill_size * 2) // 3)) if type(i) == str else i for i in img_path_list]
    res_w, res_h = img_list[0].size
    res_img = Image.new(mode='RGB', size=size, color=0)
    res_img.paste(img_list[0], box=(0, 0))
    res_img.paste(img_list[1], box=(0, res_h + fill_size))
    res_img.paste(img_list[2], box=(0, (res_h + fill_size) * 2))
    return res_img


if __name__ == '__main__':
    try:
        args = sys.argv[1:]
        output_path = args.pop(-1)
        img_path_list = args
        img_count = len(img_path_list)
        assert 2 <= img_count <= 7
        if img_count == 2:
            res_img = parser_2_h(img_path_list, size=(1005, 1005), fill_size=3)

            # img_0 = cv2.imread(img_path_list[0])
            # img_1 = cv2.imread(img_path_list[1])
            # ratio_0 = img_0.shape[0] / img_0.shape[1]
            # ratio_1 = img_1.shape[0] / img_1.shape[1]
            # if (ratio_0 >= 1.0) and ((ratio_0 - 1.0) > (1.0 - ratio_1)):
            #     img_0 = cv2.resize(img_0, (1005, 501))
            #     img_1 = cv2.resize(img_1, (1005, 501))
            #     a = np.zeros((3, 1005, 3), np.uint8)
            #     a.fill(255)
            #     img = np.hstack((img_0, a, img_1))
            #
            # else:
            #     img_0 = cv2.resize(img_0, (501, 1005))
            #     img_1 = cv2.resize(img_1, (501, 1005))
            #     a = np.zeros((1005, 3, 3), np.uint8)
            #     a.fill(255)
            #     img = np.hstack((img_0, a, img_1))
        elif img_count == 3:
            res_img_01 = parser_2_h(img_path_list[:2], size=(1005, 501), fill_size=3)
            res_img = parser_2_v([img_path_list[2], res_img_01], size=(1005, 1005), fill_size=3)

            # img_0 = cv2.imread(img_path_list[0])
            # img_1 = cv2.imread(img_path_list[1])
            # img_2 = cv2.imread(img_path_list[2])
            #
            # img_1 = cv2.resize(img_1, (501, 501))
            # img_2 = cv2.resize(img_2, (501, 501))
            # a = np.zeros((501, 3, 3), np.uint8)
            # a.fill(255)
            # img = np.hstack((img_1, a, img_2))
            #
            # img_0 = cv2.resize(img_0, (1005, 501))
            # a = np.zeros((3, 1005, 3), np.uint8)
            # a.fill(255)
            # img = np.vstack((img_0, a, img))
        elif img_count == 4:
            res_img_01 = parser_2_h(img_path_list[:2], size=(1005, 501), fill_size=3)
            res_img_23 = parser_2_h(img_path_list[2:], size=(1005, 501), fill_size=3)
            res_img = parser_2_v([res_img_01, res_img_23], size=(1005, 1005), fill_size=3)

            # img_0 = cv2.imread(img_path_list[0])
            # img_1 = cv2.imread(img_path_list[1])
            # img_2 = cv2.imread(img_path_list[2])
            # img_3 = cv2.imread(img_path_list[3])
            #
            # img_0 = cv2.resize(img_0, (501, 501))
            # img_1 = cv2.resize(img_1, (501, 501))
            # img_2 = cv2.resize(img_2, (501, 501))
            # img_3 = cv2.resize(img_3, (501, 501))
            # a = np.zeros((501, 3, 3), np.uint8)
            # a.fill(255)
            # img_0_1 = np.hstack((img_0, a, img_1))
            # img_2_3 = np.hstack((img_2, a, img_3))
            # a = np.zeros((3, 1005, 3), np.uint8)
            # a.fill(255)
            # img = np.vstack((img_0_1, a, img_2_3))
        elif img_count == 5:
            res_img_01 = parser_2_h(img_path_list[:2], size=(1005, 501), fill_size=3)
            res_img_23 = parser_2_h(img_path_list[2:4], size=(1005, 501), fill_size=3)
            res_img = parser_3_v([img_path_list[4], res_img_01, res_img_23], size=(1005, 501), fill_size=3)

            # img_0 = cv2.imread(img_path_list[0])
            # img_1 = cv2.imread(img_path_list[1])
            # img_2 = cv2.imread(img_path_list[2])
            # img_3 = cv2.imread(img_path_list[3])
            #
            # img_0 = cv2.resize(img_0, (501, 332))
            # img_1 = cv2.resize(img_1, (501, 332))
            # img_2 = cv2.resize(img_2, (501, 332))
            # img_3 = cv2.resize(img_3, (501, 332))
            # a = np.zeros((332, 3, 3), np.uint8)
            # a.fill(255)
            # img_0_1 = np.hstack((img_0, a, img_1))
            # img_2_3 = np.hstack((img_2, a, img_3))
            # a = np.zeros((3, 1005, 3), np.uint8)
            # a.fill(255)
            # img = np.vstack((img_0_1, a, img_2_3))
            #
            # img_4 = cv2.imread(img_path_list[4])
            # img_4 = cv2.resize(img_4, (1005, 501))
            # a = np.zeros((3, 1005, 3), np.uint8)
            # a.fill(255)
            # img = np.vstack((img_4, a, img))
        elif img_count == 6:
            res_img_012 = parser_3_v(img_path_list[:3], size=(332, 1005), fill_size=2)
            res_img_34 = parser_2_h(img_path_list[3:5], size=(666, 332), fill_size=2)

            img_5 = read_img(img_path_list[5]).resize((666, 666))
            res_img_345 = Image.new(mode='RGB', size=(666, 1005), color=0)
            res_img_345.paste(res_img_34, box=(0, 0))
            res_img_345.paste(img_5, box=(0, 334))

            res_img = Image.new(mode='RGB', size=(1005, 1005), color=0)
            res_img.paste(res_img_012, box=(0, 0))
            res_img.paste(res_img_345, box=(334, 0))

            # img_0 = cv2.imread(img_path_list[0])
            # img_1 = cv2.imread(img_path_list[1])
            # img_2 = cv2.imread(img_path_list[2])
            # img_3 = cv2.imread(img_path_list[3])
            # img_4 = cv2.imread(img_path_list[4])
            # img_5 = cv2.imread(img_path_list[5])
            #
            # img_0 = cv2.resize(img_0, (332, 332))
            # img_1 = cv2.resize(img_1, (332, 332))
            # img_2 = cv2.resize(img_2, (332, 332))
            # a = np.zeros((332, 3, 3), np.uint8)
            # a.fill(255)
            # img_0_1_2 = np.hstack((img_0, a, img_1, a, img_2))
            #
            # img_3 = cv2.resize(img_3, (332, 332))
            # img_4 = cv2.resize(img_4, (332, 332))
            # a = np.zeros((3, 332, 3), np.uint8)
            # a.fill(255)
            # img_3_4 = np.vstack((img_3, a, img_4))
            #
            # img_5 = cv2.resize(img_5, (667, 667))
            # a = np.zeros((667, 3, 3), np.uint8)
            # a.fill(255)
            # img_345 = np.hstack((img_3_4, a, img_5))
            #
            # a = np.zeros((3, 1002, 3), np.uint8)
            # a.fill(255)
            # img = np.vstack((img_0_1_2, a, img_345))
        elif img_count == 7:
            img_0 = read_img(img_path_list[0]).resize((669, 333))
            img_1 = read_img(img_path_list[1]).resize((333, 333))
            res_img_01 = Image.new(mode='RGB', size=(1005, 333), color=0)
            res_img_01.paste(img_0, box=(0, 0))
            res_img_01.paste(img_1, box=(669, 0))

            res_img_234 = parser_3_v(img_path_list[2:5], size=(333, 1005), fill_size=3)

            img_5 = read_img(img_path_list[5]).resize((669, 669))
            img_6 = read_img(img_path_list[6]).resize((669, 333))
            res_img_56 = Image.new(mode='RGB', size=(669, 1005), color=0)
            res_img_56.paste(img_5, box=(0, 0))
            res_img_56.paste(img_6, box=(0, 672))

            res_img = Image.new(mode='RGB', size=(1005, 1341), color=0)
            res_img.paste(res_img_01, box=(0, 0))
            res_img.paste(res_img_234, box=(0, 336))
            res_img.paste(res_img_56, box=(336, 336))

        res_img.save(output_path)
        # cv2.imwrite(output_path, img)
        sys.stdout.write('生成图片成功: {}'.format(output_path))
    except:
        sys.stdout.write('生成图片失败')
