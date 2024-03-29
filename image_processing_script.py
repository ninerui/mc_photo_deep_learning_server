# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import time
import uuid
import json
import shutil
import pickle
import logging
import threading
from random import choice

import cv2
import requests
import imagehash
import numpy as np
import tensorflow as tf
from PIL import ImageFile, Image

import conf
from utils import connects, util, image_tools
from dl_module import face_cluster_interface
from dl_module import face_emotion_interface
from dl_module import image_making_interface, face_detection_interface, face_recognition_interface
from dl_module import image_enhancement_interface
from dl_module.zhouwen_detect_blur import detection_blur
from dl_module import zhouwen_image_card_classify_interface
from dl_module.image_local_color_interface import ImageLocalColor
from dl_module.image_autocolor_interface import ImageAutoColor
from dl_module import object_detection_interface
from dl_module.id_card_detection_interface import IDCardDetection
from dl_module.image_quality_assessment_interface import TechnicalQualityModelWithTF, AestheticQualityModelWithTF

ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_restart(sleep_time):
    reboot_code = redis_connect.get(local_ip)
    if reboot_code == '1':
        logging.info('发现服务需要重启, 重启代码: {}'.format(reboot_code))
        raise SystemExit
    time.sleep(sleep_time)


class BaseThread(threading.Thread):
    """基础的线程"""
    pass


class FilePathConf:
    def __init__(self, user_id):
        self.oss_running_file = "face_cluster_data/{}/.running".format(user_id)
        self.oss_suc_img_list_file = "face_cluster_data/{}/suc_img_list.pkl".format(user_id)
        oss_connect.get_set_object(self.oss_suc_img_list_file, [])
        self.oss_face_id_with_label_file = "face_cluster_data/{}/face_id_with_label.pkl".format(user_id)
        oss_connect.get_set_object(self.oss_face_id_with_label_file, {})
        self.oss_face_data_file = "face_cluster_data/{}/face_data.pkl".format(user_id)
        oss_connect.get_set_object(self.oss_face_data_file, [])


def image_resize(image):
    m = min(image.shape[0], image.shape[1])
    f = 320.0 / m
    if f < 1.0:
        image = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))
    return image, min(f, 1.)


def call_url_func(callback_url, data_json):
    call_count = 0
    call_suc_status = False
    while call_count < 20:
        try:
            call_res = requests.post(callback_url, json=data_json)
            logging.info("回调地址: {}, 回调结果: {}".format(callback_url, call_res.text))
            if int(call_res.status_code) == 200:
                call_suc_status = True
                return call_suc_status
            else:
                time.sleep(9)
                call_count += 1
        except requests.exceptions.ConnectionError as e:
            logging.error("回调地址: {}, 错误信息: {}".format(callback_url, e))
            call_count += 1
            time.sleep(9)
            continue
        except Exception as e:
            logging.exception(e)
            call_count += 1
            time.sleep(9)
            continue
    return call_suc_status


class FaceClusterThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, thread_name):
        threading.Thread.__init__(self)
        self.thread_name = thread_name

    def main_func(self, fe_detection, fr_arcface):
        face_user_key = redis_connect.rpop(conf.redis_face_info_key_list)
        if not face_user_key:
            time.sleep(1)
            return
        face_status = face_user_key + '_status'
        redis_status = redis_connect.get(face_status)
        if redis_status == "0":  # 正在运行
            redis_connect.lpush(conf.redis_face_info_key_list, face_user_key)
            time.sleep(1)
            return
        if redis_status is None:
            redis_connect.lpush(conf.redis_face_info_key_list, face_user_key)
            redis_connect.set(face_status, '1')
            time.sleep(1)
            return
        if (redis_connect.llen(face_user_key) <= 10) and (int(redis_status) < 50):
            redis_connect.lpush(conf.redis_face_info_key_list, face_user_key)
            redis_connect.set(face_status, int(redis_status) + 1)
            time.sleep(1)
            return
        redis_connect.set(face_status, '0')
        redis_connect.srem(conf.redis_face_info_key_set, face_user_key)
        user_id = face_user_key.split('-')[1]
        try:
            face_data = []
            success_image_set = set()
            logging.info("开始取数据: {}".format(face_user_key))
            while True:
                data_ = redis_connect.rpop(face_user_key)
                if not data_:
                    logging.info("数据取完: {}".format(face_user_key))
                    break
                data_ = json.loads(data_)
                media_id = data_.get('media_id', None)

                warped = np.array(data_.get('face_data'), dtype=np.float32)
                emotion_label_arg = fe_detection.detection_emotion(warped)
                warped = np.transpose(warped, (2, 0, 1))
                emb = fr_arcface.get_feature(warped)
                data_['face_feature'] = np.array(emb).tolist()
                data_['emotionStr'] = emotion_label_arg

                face_data.append({
                    'face_feature': np.array(emb).tolist(),
                    'face_id': data_.get('face_id'),
                    "face_box": data_.get('face_box'),
                    'emotionStr': emotion_label_arg,
                })
                success_image_set.add(media_id)
            if len(face_data) > 0:
                oss_suc_img_list_file = "face_cluster_data/{}/suc_img_list.pkl".format(user_id)
                oss_connect.get_set_object(oss_suc_img_list_file, set())
                oss_face_id_with_label_file = "face_cluster_data/{}/face_id_with_label.pkl".format(user_id)
                oss_connect.get_set_object(oss_face_id_with_label_file, dict())
                oss_face_data_file = "face_cluster_data/{}/face_data.pkl".format(user_id)
                oss_connect.get_set_object(oss_face_data_file, list())

                suc_parser_img_set = pickle.loads(oss_connect.get_object(oss_suc_img_list_file).read())
                face_id_label_dict = pickle.loads(oss_connect.get_object(oss_face_id_with_label_file).read())
                old_data = pickle.loads(oss_connect.get_object(oss_face_data_file).read())

                face_data = old_data + face_data
                start_cluster_time = time.time()
                call_res_dict, face_id_label_dict = face_cluster_interface.cluster_face_func(
                    face_data, user_id, face_id_label_dict)
                logging.info('用户ID: {}, 聚类人脸耗时: {}'.format(user_id, time.time() - start_cluster_time))

                # 回调下载成功列表
                handle_result_url = choice(handle_result_url_list)
                logging.info(
                    "用户ID: {}, 开始回调 {} 结果, 共 {} 条数据...".format(
                        user_id, handle_result_url, len(call_res_dict)))
                call_results_status = call_url_func(handle_result_url, data_json={
                    'userId': user_id,
                    'content': call_res_dict
                })
                if not call_results_status:
                    # 回调失败, 保存结果
                    logging.error("用户ID: {}, 结果列表回调失败, 保存结果至oss!".format(user_id))
                    oss_connect.put_object(
                        "face_cluster_call_error_data/{}/call_result_error_{}.pkl".format(user_id, util.get_str_time()),
                        pickle.dumps({
                            "handle_result_url": handle_result_url,
                            "user_id": user_id,
                            "call_res_dict": call_res_dict,
                        }))
                oss_connect.put_object(oss_face_id_with_label_file, pickle.dumps(face_id_label_dict))
                oss_connect.put_object(oss_face_data_file, pickle.dumps(face_data))
                oss_connect.put_object(oss_suc_img_list_file, pickle.dumps(success_image_set | set(suc_parser_img_set)))
        except Exception as e:
            logging.exception(e)
            return
        finally:
            redis_connect.delete(face_status)
            return

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        fr_arcface = face_recognition_interface.FaceRecognitionWithArcFace()
        fe_detection = face_emotion_interface.FaceEmotionKeras()  # 表情检测模型, 不能跨线程
        logging.info("人脸聚类线程已启动...")
        while True:
            try:
                self.main_func(fe_detection, fr_arcface)
            except Exception as e:
                logging.exception(e)
                continue
            finally:
                check_restart(2)


def get_redis_next_data(rds_name):
    params_data = redis_connect.rpop(rds_name)
    if params_data:
        params = json.loads(params_data)
        logging.info("剩余数据: {}, data: {}".format(redis_connect.llen(conf.redis_image_making_list_name), params))

        image_url = params.get("image_url")
        receiving_data_type = params.get('data_type')
        if int(receiving_data_type) == 21:  # 有看模块, 进行相似和质量, image_url为oss key
            local_image_path = os.path.join(conf.tmp_image_dir, '21__' + os.path.basename(image_url))
            local_save_image_path = os.path.join(conf.tmp_youkan_image_dir, os.path.basename(image_url))
            writing_oss_connect.get_object_to_file(image_url, local_image_path)
            shutil.copy(local_image_path, local_save_image_path)
            download_code = 1
            image_path = local_image_path
        else:  # 相册打标签等全功能
            res_data = image_tools.download_and_parser_image(image_url, conf.tmp_image_dir)
            download_code = res_data.get('code')
            image_path = res_data.get('image_path')

        if download_code == -1:  # 下载失败, 打回列表
            reg_count = params.get('reg_count', 0)
            if reg_count > 100:
                redis_connect.lpush(
                    conf.redis_image_making_error_name,
                    json.dumps({'error_type': "download_fail", "error_data": params}))
                return None
            params['reg_count'] = reg_count + 1
            time.sleep(2)
            redis_connect.rpush(conf.redis_image_making_list_name, json.dumps(params))
        elif download_code == 1:  # 下载以及解析成功
            params['image_path'] = image_path
            return params
        elif download_code == 3:
            logging.error("图片占用内存为0, params data: {}".format(params))
            redis_connect.srem(conf.redis_image_making_set_name, params.get('media_id'))
            return None
        # elif download_code == 2:  # 下载以及解析成功, 图片格式为mp4
        #     params['image_path'] = image_path
        #     return params
        else:  # 图片处理失败
            oss_key = "error_image/{}".format(os.path.basename(image_path))
            redis_connect.lpush(
                conf.redis_image_making_error_name,
                json.dumps({
                    'error_code': download_code,
                    "img_type": res_data.get('img_type'),
                    "error_data": params,
                    "oss_key": oss_key,
                    "error_info": res_data.get('info'),
                }))
            oss_connect.put_object_from_file(oss_key, image_path)
            util.removefile(image_path)
            data_json = {
                'mediaId': params.get('media_id'),
                'fileId': params.get('file_id'),
                'tag': str([]),
                'filePath': image_url,
                'exponent': 0,
                'mediaInfo': str(json.dumps({
                    "certificateInfo": [],
                    "quality": [],
                    "human_coordinate": "",
                }, ensure_ascii=False)),
                "isBlackAndWhite": 0,
                "isLocalColor": 0,
                'existFace': 0,
            }
            call_url_func(params.get('callback_url'), data_json=data_json)
    return None


def get_is_black_and_white(img):
    """
    获取图片是否符合黑白的特征
    :param img: 输入的图像
    :return: bool值
    """
    b, g, r = cv2.split(img)
    all_pixel = b.size
    if (np.count_nonzero(b == g) / all_pixel > 0.99) and (np.count_nonzero(b == r) / all_pixel > 0.99) and (
            np.count_nonzero(r == g) / all_pixel > 0.99) and len(set(img.flatten())) >= 80:
        return 1
    return 0


class ImageProcessingThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, thread_name):
        threading.Thread.__init__(self)
        self.thread_name = thread_name

    def parser_face(self, user_id, media_id, image):
        face_count = 0
        try:
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_height, im_width = image.shape[:2]
            f = min((4096. / max(image.shape[0], image.shape[1])), 1.0)
            image_r = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))

            image_np_expanded = np.expand_dims(image_r, axis=0)

            fd_boxes_, fd_scores_ = fd_ssd_detection.detect_face(image_np_expanded)

            for idx in range(fd_boxes_[0].shape[0]):
                if fd_scores_[0][idx] < 0.7:
                    break
                ymin, xmin, ymax, xmax = fd_boxes_[0][idx]
                add_y_border = (ymax - ymin) * 0.1
                add_x_border = (xmax - xmin) * 0.1
                xmin_, xmax_ = max(0, xmin - add_x_border), min(1, xmax + add_x_border)
                ymin_, ymax_ = max(0, ymin - add_y_border), min(1, ymax + add_y_border)
                left, right, top, bottom = map(
                    int, (xmin_ * im_width, xmax_ * im_width, ymin_ * im_height, ymax_ * im_height))

                face_image = image_np[top:bottom, left:right, :]
                if max(face_image.shape[:2]) < 56.:
                    continue
                face_image_resize, im_scale = image_resize(face_image)
                mtcnn_res = fd_mtcnn_detection.detect_face(face_image_resize)
                if not mtcnn_res:
                    continue
                mtcnn_scare = mtcnn_res['confidence']
                if mtcnn_scare < 0.96:
                    continue
                mtcnn_box = mtcnn_res['box']
                if max(mtcnn_box[2:]) < 50.:
                    continue
                mtcnn_points = mtcnn_res['keypoints']
                mtcnn_points = np.asarray([
                    [mtcnn_points['left_eye'][0] / im_scale + left, mtcnn_points['left_eye'][1] / im_scale + top],
                    [mtcnn_points['right_eye'][0] / im_scale + left, mtcnn_points['right_eye'][1] / im_scale + top],
                    [mtcnn_points['nose'][0] / im_scale + left, mtcnn_points['nose'][1] / im_scale + top],
                    [mtcnn_points['mouth_left'][0] / im_scale + left, mtcnn_points['mouth_left'][1] / im_scale + top],
                    [mtcnn_points['mouth_right'][0] / im_scale + left, mtcnn_points['mouth_right'][1] / im_scale + top],
                ])
                warped = image_tools.preprocess(image_np, [112, 112], bbox=mtcnn_box, landmark=mtcnn_points)
                face_image_path = os.path.join(
                    conf.tmp_image_dir, "{}_{}.jpg".format(media_id, idx))
                cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                oss_face_image_name = "face_cluster_data/{}/face_images/{}_{}.jpg".format(
                    user_id, media_id, idx)
                oss_connect.put_object_from_file(oss_face_image_name, face_image_path)
                util.removefile(face_image_path)

                redis_user_key = conf.redis_face_info_name.format(user_id)
                if redis_connect.sadd(conf.redis_face_info_key_set, redis_user_key) == 1:
                    redis_connect.lpush(conf.redis_face_info_key_list, redis_user_key)
                # if redis_connect.llen(redis_user_key) == 0:
                #     r_set_code = redis_connect.sadd(conf.redis_face_info_key_set, redis_user_key)
                redis_connect.lpush(redis_user_key, json.dumps({
                    "media_id": media_id,
                    "face_id": "{}_{}".format(media_id, idx),
                    "face_box": [left, top, right - left, bottom - top],
                    "face_data": warped.tolist(),
                }))
                face_count += 1
        except Exception as e:
            logging.exception("{}上传失败\n{}".format(media_id, e))
        finally:
            return face_count

    def get_is_local_color(self, image, face_count):
        res = 1
        tiaojian = False
        location = ""
        try:
            f = min((4096. / max(image.shape[0], image.shape[1])), 1.0)
            image_r = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))
            image_np_expanded = np.expand_dims(image_r, axis=0)
            od_boxes, od_scores, od_classes = object_detection_model.detect_object(image_np_expanded)
            if face_count == 1:
                ymin, xmin, ymax, xmax = od_boxes[0][0]
                if (abs((xmax - xmin) / 2. + xmin - 0.5) < 0.1) and (ymin < 0.4) and (ymax > 0.6):
                    tiaojian = True
                location = "{}{}{}{}".format(int(ymin * 10), int(xmin * 10), int(ymax * 10), int(xmax * 10))
            else:
                for idx in range(od_boxes[0].shape[0]):
                    if od_scores[0][idx] < 0.7:
                        break
                    if int(od_classes[0][idx]) != 1:
                        continue
                    res += 1
                    ymin, xmin, ymax, xmax = od_boxes[0][idx]
                    if (abs((xmax - xmin) / 2. + xmin - 0.5) < 0.1) and (ymin < 0.4) and (ymax > 0.6):
                        tiaojian = True
        except Exception as e:
            logging.exception(e)
        finally:
            if res == 1 and tiaojian:
                return 1, location
            else:
                return 0, location

    # def get_face_info_from_image(self, image):
    #     face_info_list = []
    #     try:
    #         image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         im_height, im_width = image.shape[:2]
    #         f = min((4096. / max(image.shape[0], image.shape[1])), 1.0)
    #         image_r = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))
    #         image_np_expanded = np.expand_dims(image_r, axis=0)
    #         fd_boxes_, fd_scores_ = fd_ssd_detection.detect_face(image_np_expanded)
    #         for idx in range(fd_boxes_[0].shape[0]):
    #             if fd_scores_[0][idx] < 0.7:
    #                 break
    #             ymin, xmin, ymax, xmax = fd_boxes_[0][idx]
    #             add_y_border = (ymax - ymin) * 0.1
    #             add_x_border = (xmax - xmin) * 0.1
    #             xmin_, xmax_ = max(0, xmin - add_x_border), min(1, xmax + add_x_border)
    #             ymin_, ymax_ = max(0, ymin - add_y_border), min(1, ymax + add_y_border)
    #             left, right, top, bottom = map(
    #                 int, (xmin_ * im_width, xmax_ * im_width, ymin_ * im_height, ymax_ * im_height))
    #
    #             face_image = image_np[top:bottom, left:right, :]
    #             if max(face_image.shape[:2]) < 56.:
    #                 continue
    #             face_image_resize, im_scale = image_resize(face_image)
    #             mtcnn_res = fd_mtcnn_detection.detect_face(face_image_resize)
    #             if not mtcnn_res:
    #                 continue
    #             mtcnn_scare = mtcnn_res['confidence']
    #             if mtcnn_scare < 0.96:
    #                 continue
    #             mtcnn_box = mtcnn_res['box']
    #             if max(mtcnn_box[2:]) < 50.:
    #                 continue
    #             mtcnn_points = mtcnn_res['keypoints']
    #             mtcnn_points = np.asarray([
    #                 [mtcnn_points['left_eye'][0] / im_scale + left, mtcnn_points['left_eye'][1] / im_scale + top],
    #                 [mtcnn_points['right_eye'][0] / im_scale + left, mtcnn_points['right_eye'][1] / im_scale + top],
    #                 [mtcnn_points['nose'][0] / im_scale + left, mtcnn_points['nose'][1] / im_scale + top],
    #                 [mtcnn_points['mouth_left'][0] / im_scale + left, mtcnn_points['mouth_left'][1] / im_scale + top],
    #                 [mtcnn_points['mouth_right'][0] / im_scale + left, mtcnn_points['mouth_right'][1] / im_scale + top],
    #             ])
    #             warped = image_tools.preprocess(image_np, [112, 112], bbox=mtcnn_box, landmark=mtcnn_points)
    #
    #
    #
    #             face_image_path = os.path.join(
    #                 conf.tmp_image_dir, "{}_{}.jpg".format(media_id, idx))
    #             cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    #             oss_face_image_name = "face_cluster_data/{}/face_images/{}_{}.jpg".format(
    #                 user_id, media_id, idx)
    #             oss_connect.put_object_from_file(oss_face_image_name, face_image_path)
    #             util.removefile(face_image_path)
    #
    #             redis_user_key = conf.redis_face_info_name.format(user_id)
    #             if redis_connect.sadd(conf.redis_face_info_key_set, redis_user_key) == 1:
    #                 redis_connect.lpush(conf.redis_face_info_key_list, redis_user_key)
    #             # if redis_connect.llen(redis_user_key) == 0:
    #             #     r_set_code = redis_connect.sadd(conf.redis_face_info_key_set, redis_user_key)
    #             redis_connect.lpush(redis_user_key, json.dumps({
    #                 "media_id": media_id,
    #                 "face_id": "{}_{}".format(media_id, idx),
    #                 "face_box": [left, top, right - left, bottom - top],
    #                 "face_data": warped.tolist(),
    #             }))
    #             face_count += 1
    #     except Exception as e:
    #         logging.exception("{}上传失败\n{}".format(media_id, e))
    #     finally:
    #         return face_count

    def parser_picture(self, image_path, **kwargs):
        tmp_time = time.time()
        if kwargs.get('making', True):
            tag = oi_5000_model.get_tag_from_one(image_path)
            time_making = time.time() - tmp_time
            tmp_time = time.time()
        # 读取图片
        image = cv2.imread(image_path)
        if kwargs.get('ic_card_detect', True):
            have_id_card = have_idcard_model.detect_id_card(image)
            if have_id_card:
                is_card = is_idcard_model.get_res_from_one(image)
            else:
                is_card = []
            time_ic = time.time() - tmp_time

    def parser_video(self, file_path):
        videoCapture = cv2.VideoCapture(file_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        logging.info(f"video fps: {fps}, video frames: {frames}")
        if frames // fps > 50:  # 视频超过了50s，则最多取50张
            get_frame_list = [int(frames // 50) * i for i in range(50)]
        else:
            get_frame_list = [int(fps) * i for i in range(int(frames // fps))]

        label_list = []
        face_list = []
        tmp_image_hash_value_list = []
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            if i not in get_frame_list:
                continue
            image_hash_value = imagehash.dhash(Image.fromarray(frame), hash_size=8)
            if len(tmp_image_hash_value_list) == 0:  # 第一张
                tmp_image_hash_value_list.append(image_hash_value)
                img_str = cv2.imencode('.jpg', frame)[1].tostring()
                label_list = label_list + oi_5000_model.get_tag_from_one(img_str)
                continue
            if min([image_hash_value - j for j in tmp_image_hash_value_list]) <= 9:  # 相似跳过
                continue
            img_str = cv2.imencode('.jpg', frame)[1].tostring()
            label_list = label_list + oi_5000_model.get_tag_from_one(img_str)
            tmp_image_hash_value_list.append(image_hash_value)
        return list(set(label_list))

    def parser_gif(self):
        pass

    def main_func(self):
        start_time = time.time()
        params_data = get_redis_next_data(conf.redis_image_making_list_name)
        if params_data is None:
            return
        logging.info("image data: {}".format(params_data))
        image_path = params_data.get('image_path')
        receiving_data_type = params_data.get('data_type')
        if int(receiving_data_type) == 21:  # 有看模块, 进行相似和质量, image_url为oss key
            quality_value, similarity_value = aesthetic_model.get_baseline_and_res(image_path)
            # phash计算图片相似
            similarity_value = imagehash.phash(Image.open(image_path), hash_size=8, highfreq_factor=4)
            similarity_value = ''.join(str(b) for b in 1 * similarity_value.hash.flatten())  # 转成'10101010100'格式
            data_json = {
                "fileId": params_data.get("fileId"),
                "md5": params_data.get("md5"),
                "key": params_data.get("image_url"),
                "qualityValue": quality_value,
                "similarityValue": similarity_value,
            }
            logging.info("有看回调结果: {}".format(data_json))
            call_results_status = call_url_func(params_data.get('callback_url'), data_json=data_json)
            logging.info("有看回调结果: {}".format(call_results_status))
            return
        media_id = params_data.get('media_id')
        user_id = params_data.get('user_id')
        image_url = params_data.get('image_url')

        time_dl = time.time() - start_time
        # 开始图片打标
        tmp_time = time.time()
        tag = oi_5000_model.get_tag_from_one(image_path)
        time_making = time.time() - tmp_time
        # 读取图片
        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape

        # 图片证件识别
        time_ic, time_face, time_od, is_black_and_white, is_local_color, face_count = 0, 0, 0, 0, 0, 0
        is_card = []
        location = ""
        if max(img_h, img_w) / min(img_h, img_w) < 9:  # 判断长图, 比率超过33会报错
            tmp_time = time.time()
            have_id_card = have_idcard_model.detect_id_card(image)
            if have_id_card:
                top, left, bottom, right = have_id_card
                top, left, bottom, right = int(top * img_h), int(left * img_w), int(bottom * img_h), int(right * img_w)
                img_crop = image[top:bottom, left:right, :]
                is_card = is_idcard_model.get_res_from_one(img_crop)
            time_ic = time.time() - tmp_time
            # 人脸处理
            tmp_time = time.time()
            face_count = self.parser_face(user_id, media_id, image)
            time_face = time.time() - tmp_time
            tmp_time = time.time()
            is_local_color, location = self.get_is_local_color(image, face_count)
            time_od = time.time() - tmp_time

            is_black_and_white = get_is_black_and_white(image)

        quality_image = np.asarray(tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224)))
        data_json = {
            'mediaId': media_id,
            'fileId': params_data.get('file_id'),
            'tag': str(tag),
            'filePath': image_url,
            'exponent': detection_blur(image),
            'mediaInfo': str(json.dumps({
                "certificateInfo": is_card,
                "quality": [aesthetic_model.get_res(quality_image), technical_model.get_res(quality_image)],
                "human_coordinate": location,
            }, ensure_ascii=False)),
            "isBlackAndWhite": is_black_and_white,
            "isLocalColor": is_local_color,
            'existFace': min(face_count, 127),
        }
        call_results_status = call_url_func(params_data.get('callback_url'), data_json=data_json)

        if is_black_and_white == 1:  # 是黑白图片
            oss_key = "wonderful_tmp_dir/{}".format(os.path.basename(image_path))
            oss_connect.put_object_from_file(oss_key, image_path)
            redis_connect.lpush(conf.redis_wonderful_gen_name, json.dumps({
                "type": 12,
                "userId": user_id,
                "mediaId": media_id,
                "oss_key": oss_key
            }))
        util.removefile(image_path)

        redis_connect.srem(conf.redis_image_making_set_name, media_id)
        logging.info("{} total: {:.4f}, dl: {:.4f}, making: {:.4f}, ic: {:.4f}, face: {:.4f}, od: {:.4f}".format(
            os.path.basename(image_url), time.time() - start_time, time_dl, time_making, time_ic, time_face, time_od))

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        logging.info("图片解析线程已启动...")
        while True:
            try:
                self.main_func()
            except Exception as e:
                logging.exception(e)
                continue
            finally:
                check_restart(1)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


class GenerationWonderfulImageThread(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        time.sleep(0.01)

    def check_restart(self, params_count):
        reboot_code = redis_connect.get(local_ip)
        if reboot_code == '1':
            logging.info('发现服务需要重启, 重启代码: {}'.format(reboot_code))
            raise SystemExit
        time.sleep(9 / max(1, params_count))

    def download_image(self, image_url):
        res_data = image_tools.download_and_parser_image(image_url, conf.tmp_image_dir)
        download_code = res_data.get('code')
        image_path = res_data.get('image_path')
        if download_code == 1:
            return image_path
        else:
            return None

    def run(self):
        logging.info("精彩生成线程已启动...")
        while True:
            try:
                params_count = redis_connect.llen(conf.redis_wonderful_gen_name)
                params = redis_connect.rpop(conf.redis_wonderful_gen_name)
                if not params:
                    self.check_restart(params_count)
                    continue
                params = json.loads(params)
                logging.info("开始生成精彩, 剩余数据: {} 条, 当前数据: {}".format(params_count - 1, params))
                user_id = params.get("userId", None)
                callback_url = params.get('callbackUrl', choice(wonderful_callback_url_list))
                media_id = params.get("mediaId", None)
                image_local_path = params.get('imageLocalPath', None)

                wonderful_type = int(params.get("type"))
                strftime = util.get_str_time()
                img_name = "{}_{}_{}.jpg".format(str(uuid.uuid1()).replace('-', ''), wonderful_type, strftime)
                output_path = os.path.join(conf.tmp_image_dir, img_name)
                oss_image_path = "wonderful_image/{}/{}/{}".format(wonderful_type, user_id, img_name)
                tmp_time = time.time()

                if wonderful_type == 8:  # 过去现在
                    past_new_data = json.loads(params.get('pastNowData'))
                    img_data_0 = past_new_data[0]
                    img_url_0 = img_data_0.get("imgUrl")
                    img_time_0 = int(img_data_0.get("photoTime")) // 1000
                    img_human_0 = img_data_0.get("tempHumanCoordinate")
                    img_data_1 = past_new_data[1]
                    img_url_1 = img_data_1.get("imgUrl")
                    img_time_1 = int(img_data_1.get("photoTime")) // 1000
                    img_human_1 = img_data_1.get("tempHumanCoordinate")
                    image_path_0 = self.download_image(img_url_0)
                    image_path_1 = self.download_image(img_url_1)
                    image_tools.create_past_now_img(
                        [image_path_0, image_path_1], [img_time_0, img_time_1], [img_human_0, img_human_1], output_path)
                    util.removefile(image_path_0)
                    util.removefile(image_path_1)
                else:
                    image_url = params.get('imageUrl', None)
                    if image_url is not None:
                        image_path = self.download_image(image_url)
                    else:
                        oss_key = params.get("oss_key")
                        image_path = os.path.join(conf.tmp_image_dir, os.path.basename(oss_key))
                        oss_connect.get_object_to_file(oss_key, image_path)
                        oss_connect.delete_object(oss_key)
                    assert os.path.isfile(image_path)

                    if wonderful_type == 11:  # 风格化照片
                        image_enhancement_model.get_hdr_image(image_path, output_path)
                    elif wonderful_type == 12:  # 自动上色
                        autocolor_model.get_result_image(image_path, output_path)
                    elif wonderful_type == 9:  # 局部彩色
                        create_local_color.get_result(image_path, output_path)
                    elif wonderful_type == 16:  # 增强画质
                        pass
                    util.removefile(image_path)

                logging.info("wonderful_type: {}, 耗时: {}".format(wonderful_type, time.time() - tmp_time))
                oss_connect.put_object_from_file(oss_image_path, output_path)
                util.removefile(output_path)
                if int(wonderful_type) == 12:
                    continue
                call_url_func(callback_url, data_json={
                    "ossKey": oss_image_path,
                    "type": wonderful_type,
                    "oldMediaId": media_id,
                    "imageLocalPath": image_local_path,
                    "userId": user_id
                })

            except Exception as e:
                logging.exception(e)
            finally:
                self.check_restart(9)


if __name__ == '__main__':
    # 创建日志文件
    util.makedirs(conf.log_dir)
    util.makedirs(conf.tmp_image_dir)
    util.makedirs(conf.tmp_youkan_image_dir)
    py_file_name = os.path.basename(__file__).split('.')[0]
    log_file = os.path.join(conf.log_dir, "{}.log".format(py_file_name))
    util.init_logging(log_file, log_filelevel=logging.INFO, log_streamlevel=logging.INFO, daily=False)
    logging.info("开始启动程序...")
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    local_ip = util.get_local_ip()  # 获取ip
    env_code = conf.env_manage.get(local_ip, 0)
    logging.info("获取运行环境代码为: {}(0: 测试服, 1: 正式服), 开始连接oss和redis...".format(env_code))
    account_conf = conf.AccountConf(env_code=env_code)  # 获取相册的账号配置
    writing_account_conf = conf.WritingAccountConf(env_code=env_code)  # 获取有看的账号配置
    # 连接oss
    oss_connect = connects.ConnectALiYunOSS(
        account_conf.accessKeyId,
        account_conf.accessKeySecret,
        account_conf.endpoint,
        account_conf.bucket
    )  # 相册的oss链接
    writing_oss_connect = connects.ConnectALiYunOSS(
        writing_account_conf.accessKeyId,
        writing_account_conf.accessKeySecret,
        writing_account_conf.endpoint,
        writing_account_conf.bucket
    )  # 有看的oss链接
    # 连接redis
    redis_connect = connects.ConnectRedis(
        account_conf.res_host, account_conf.res_port, account_conf.res_decode_responses, account_conf.res_password
    ).r_object
    # 设置脚本重启代码
    redis_connect.set(local_ip, "0")

    logging.info("加载全局模型...")
    image_process_thread_num = conf.image_process_thread_num_dict.get(local_ip, 0)
    wonderful_gen_thread_num = conf.wonderful_gen_thread_num_dict.get(local_ip, 0)
    face_cluster_thread_num = conf.face_cluster_thread_num_dict.get(local_ip, 0)

    handle_result_url_list = conf.handle_result_url_dict.get(str(env_code))
    wonderful_callback_url_list = conf.wonderful_callback_url_dict.get(str(env_code))

    if image_process_thread_num > 0:
        oi_5000_model = image_making_interface.ImageMakingWithOpenImage()
        fd_ssd_detection = face_detection_interface.FaceDetectionWithSSDMobilenet()
        fd_mtcnn_detection = face_detection_interface.FaceDetectionWithMtcnnTF(steps_threshold=[0.6, 0.7, 0.8])
        is_idcard_model = zhouwen_image_card_classify_interface.IDCardClassify(
            # model_path='./models/zhouwen_models/id_card_classifly_v1.2.pb'
        )
        object_detection_model = object_detection_interface.ObjectDetectionWithSSDMobilenetV2()
        have_idcard_model = IDCardDetection()
        technical_model = TechnicalQualityModelWithTF()
        aesthetic_model = AestheticQualityModelWithTF()

    if wonderful_gen_thread_num > 0:
        autocolor_model = ImageAutoColor()
        image_enhancement_model = image_enhancement_interface.ImageHDRs()
        create_local_color = ImageLocalColor()

    # 创建线程并开始图片打标线程
    for i in range(image_process_thread_num):
        ImageProcessingThread("Thread_{}".format(i)).start()

    # 创建线程并开始人脸聚类线程
    for i in range(face_cluster_thread_num):
        FaceClusterThread("face_cluster_{}".format(i)).start()

    # 创建并开始精彩生成线程
    for i in range(wonderful_gen_thread_num):
        GenerationWonderfulImageThread("wonderful_generation_{}".format(i)).start()

    while True:
        active_thread_count = threading.active_count()
        if active_thread_count == 1:
            logging.info('所有线程已停止, 等待重启...')
            break
        time.sleep(3)
