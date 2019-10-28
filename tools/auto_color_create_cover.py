# !/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
import sys
import traceback

from PIL import Image


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


if __name__ == '__main__':
    try:
        args = sys.argv[1:]
        output_path = args.pop(-1)
        img_path_list = args
        img_count = len(img_path_list)
        assert 2 <= img_count <= 9
        if img_count == 2:
            image_list = [Image.open(i) for i in img_path_list]
            image_direction_list = [(i.size[0] / i.size[1]) - 1. for i in image_list]
            if sum(image_direction_list) > 0:
                img_1 = image_list.pop(0)
                img_1 = resize_image(img_1, size=(1025, 511))
                img_2 = image_list.pop(0)
                img_2 = resize_image(img_2, size=(1025, 511))
                res_img = Image.new(mode='RGB', size=(1025, 1025), color='white')
                res_img.paste(img_1, box=(0, 0))
                res_img.paste(img_2, box=(0, 514))
            else:
                img_1 = image_list.pop(0)
                img_1 = resize_image(img_1, size=(511, 1025))
                img_2 = image_list.pop(0)
                img_2 = resize_image(img_2, size=(511, 1025))
                res_img = Image.new(mode='RGB', size=(1025, 1025), color='white')
                res_img.paste(img_1, box=(0, 0))
                res_img.paste(img_2, box=(514, 0))

        elif img_count == 3:
            image_list = [Image.open(i) for i in img_path_list]
            image_direction_list = [(i.size[0] - i.size[1]) for i in image_list]

            hx_0_idx = image_direction_list.index(max(image_direction_list))
            image_direction_list.pop(hx_0_idx)
            img_0 = image_list.pop(hx_0_idx)
            img_0 = resize_image(img_0, size=(1025, 511))

            img_1 = image_list.pop(0)
            img_1 = resize_image(img_1, size=(511, 511))

            img_2 = image_list.pop(0)
            img_2 = resize_image(img_2, size=(511, 511))

            res_img = Image.new(mode='RGB', size=(1025, 1025), color='white')
            res_img.paste(img_0, box=(0, 0))
            res_img.paste(img_1, box=(0, 514))
            res_img.paste(img_2, box=(514, 514))

        elif img_count == 4:
            image_list = [Image.open(i) for i in img_path_list]

            img_1 = image_list.pop(0)
            img_1 = resize_image(img_1, size=(511, 511))

            img_2 = image_list.pop(0)
            img_2 = resize_image(img_2, size=(511, 511))

            img_3 = image_list.pop(0)
            img_3 = resize_image(img_3, size=(511, 511))

            img_4 = image_list.pop(0)
            img_4 = resize_image(img_4, size=(511, 511))

            res_img = Image.new(mode='RGB', size=(1025, 1025), color='white')
            res_img.paste(img_1, box=(0, 514))
            res_img.paste(img_2, box=(0, 0))
            res_img.paste(img_3, box=(514, 514))
            res_img.paste(img_4, box=(514, 0))

        elif img_count == 5:
            image_list = [Image.open(i) for i in img_path_list]
            image_direction_list = [(i.size[0] - i.size[1]) for i in image_list]

            hx_0_idx = image_direction_list.index(max(image_direction_list))
            image_direction_list.pop(hx_0_idx)
            img_0 = image_list.pop(hx_0_idx)
            img_0 = resize_image(img_0, size=(1025, 511))

            img_1 = image_list.pop(0)
            img_1 = resize_image(img_1, size=(511, 511))

            img_2 = image_list.pop(0)
            img_2 = resize_image(img_2, size=(511, 511))

            img_3 = image_list.pop(0)
            img_3 = resize_image(img_3, size=(511, 511))

            img_4 = image_list.pop(0)
            img_4 = resize_image(img_4, size=(511, 511))

            res_img = Image.new(mode='RGB', size=(1025, 1539), color='white')
            res_img.paste(img_0, box=(0, 0))
            res_img.paste(img_1, box=(0, 514))
            res_img.paste(img_2, box=(0, 1028))
            res_img.paste(img_3, box=(514, 514))
            res_img.paste(img_4, box=(514, 1028))

        elif img_count == 6:
            image_list = [Image.open(i) for i in img_path_list]

            img_0 = image_list.pop(0)
            img_0 = resize_image(img_0, size=(1025, 1025))

            img_1 = image_list.pop(0)
            img_1 = resize_image(img_1, size=(511, 511))

            img_2 = image_list.pop(0)
            img_2 = resize_image(img_2, size=(511, 511))

            img_3 = image_list.pop(0)
            img_3 = resize_image(img_3, size=(511, 511))

            img_4 = image_list.pop(0)
            img_4 = resize_image(img_4, size=(511, 511))

            img_5 = image_list.pop(0)
            img_5 = resize_image(img_5, size=(511, 511))

            res_img = Image.new(mode='RGB', size=(1539, 1539), color='white')
            res_img.paste(img_0, box=(514, 514))
            res_img.paste(img_1, box=(0, 0))
            res_img.paste(img_2, box=(0, 514))
            res_img.paste(img_3, box=(0, 1028))
            res_img.paste(img_4, box=(514, 0))
            res_img.paste(img_5, box=(1028, 0))

        elif img_count == 7:
            image_list = [Image.open(i) for i in img_path_list]
            image_direction_list = [(i.size[0] - i.size[1]) for i in image_list]

            hx_0_idx = image_direction_list.index(max(image_direction_list))
            image_direction_list.pop(hx_0_idx)
            img_0 = image_list.pop(hx_0_idx)
            img_0 = resize_image(img_0, size=(1025, 511))

            hx_1_idx = image_direction_list.index(max(image_direction_list))
            image_direction_list.pop(hx_1_idx)
            img_1 = image_list.pop(hx_1_idx)
            img_1 = resize_image(img_1, size=(1025, 511))

            img_2 = image_list.pop(0)
            img_2 = resize_image(img_2, size=(1025, 1025))

            img_3 = image_list.pop(0)
            img_3 = resize_image(img_3, size=(511, 511))

            img_4 = image_list.pop(0)
            img_4 = resize_image(img_4, size=(511, 511))

            img_5 = image_list.pop(0)
            img_5 = resize_image(img_5, size=(511, 511))

            img_6 = image_list.pop(0)
            img_6 = resize_image(img_6, size=(511, 511))

            res_img = Image.new(mode='RGB', size=(1539, 2053), color='white')
            res_img.paste(img_0, box=(0, 0))
            res_img.paste(img_1, box=(514, 1542))
            res_img.paste(img_2, box=(514, 514))
            res_img.paste(img_3, box=(1028, 0))
            res_img.paste(img_4, box=(0, 514))
            res_img.paste(img_5, box=(0, 1028))
            res_img.paste(img_6, box=(0, 1542))

        elif img_count == 9:
            image_list = [Image.open(i) for i in img_path_list]

            img_1 = image_list.pop(0)
            img_1 = resize_image(img_1, size=(511, 511))

            img_2 = image_list.pop(0)
            img_2 = resize_image(img_2, size=(511, 511))

            img_3 = image_list.pop(0)
            img_3 = resize_image(img_3, size=(511, 511))

            img_4 = image_list.pop(0)
            img_4 = resize_image(img_4, size=(511, 511))

            img_5 = image_list.pop(0)
            img_5 = resize_image(img_5, size=(511, 511))

            img_6 = image_list.pop(0)
            img_6 = resize_image(img_6, size=(511, 511))

            img_7 = image_list.pop(0)
            img_7 = resize_image(img_7, size=(511, 511))

            img_8 = image_list.pop(0)
            img_8 = resize_image(img_8, size=(511, 511))

            img_9 = image_list.pop(0)
            img_9 = resize_image(img_9, size=(511, 511))

            res_img = Image.new(mode='RGB', size=(1539, 1539), color='white')
            res_img.paste(img_1, box=(0, 0))
            res_img.paste(img_2, box=(0, 514))
            res_img.paste(img_3, box=(0, 1028))
            res_img.paste(img_4, box=(514, 0))
            res_img.paste(img_5, box=(1028, 0))
            res_img.paste(img_6, box=(514, 514))
            res_img.paste(img_7, box=(514, 1028))
            res_img.paste(img_8, box=(1028, 514))
            res_img.paste(img_9, box=(1028, 1028))

        res_img.save(output_path)
        sys.stdout.write('111: {}'.format(output_path))
    except Exception as e:
        sys.stdout.write('000')
        traceback.print_exc()
