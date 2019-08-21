import os
import time
import json
import pickle
# import shutil
import logging
import threading
# import collections
from urllib.request import urlretrieve

import cv2
import imageio
import requests
import numpy as np
# from tensorflow import keras
from PIL import Image, ImageFile
from PIL import ImageFilter, ImageColor

import conf
from utils import connects, util, image_tools
from dl_module import face_cluster_interface
from dl_module import face_emotion_interface
# from dl_module import image_quality_assessment_interface
from dl_module import zhouwen_image_card_classify_interface
from dl_module import image_making_interface, face_detection_interface, face_recognition_interface
from dl_module import image_enhancement_interface
# from dl_module.fasterai.visualize import get_image_colorizer
from dl_module.human_pose_estimation_interface import TfPoseEstimator
# from dl_module.object_mask_detection_interface import ObjectMaskDetection
from dl_module.zhouwen_detect_blur import detection_blur
from dl_module.image_local_color_interface import ImageLocalColor
from dl_module.image_autocolor_interface import ImageAutoColor
from dl_module import object_detection_interface

try:
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception as e1:
    print(e1)
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_restart(sleep_time):
    reboot_code = r_object.get_content(local_ip)
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
        oss_bucket.get_set_object(self.oss_suc_img_list_file, [])
        self.oss_face_id_with_label_file = "face_cluster_data/{}/face_id_with_label.pkl".format(user_id)
        oss_bucket.get_set_object(self.oss_face_id_with_label_file, {})
        self.oss_face_data_file = "face_cluster_data/{}/face_data.pkl".format(user_id)
        oss_bucket.get_set_object(self.oss_face_data_file, [])


def image_resize(image):
    m = min(image.shape[0], image.shape[1])
    f = 320.0 / m
    if f < 1.0:
        image = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))
    return image, min(f, 1.)


def call_url_func(user_id, callback_url, data_json):
    call_count = 0
    call_suc_status = False
    while call_count < 20:
        try:
            call_res = requests.post(callback_url, json=data_json)
            if int(call_res.status_code) == 200:
                call_suc_status = True
                logging.info("用户ID: {}, call_status: {}".format(user_id, call_res.text))
                return call_suc_status
            else:
                logging.error("用户ID: {}, call_status: {}".format(user_id, call_res.text))
                time.sleep(9)
                call_count += 1
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
        self.log_content = "线程名: {}".format(thread_name) + ", {}"

    def log_error(self, content):
        logging.error(self.log_content.format(content))

    def log_info(self, content):
        logging.info(self.log_content.format(content))

    def log_exception(self, content):
        logging.exception(self.log_content.format(content))

    def check_restart(self, params_count):
        reboot_code = r_object.get_content(local_ip)
        if reboot_code == '1':
            self.log_info('发现服务需要重启, 重启代码: {}'.format(reboot_code))
            raise SystemExit
        time.sleep(9 / max(1, params_count))

    def main_func(self, fe_detection, fr_arcface):
        face_user_key = r_object.rpop_content(conf.redis_face_info_key_list)
        if not face_user_key:
            return
        if r_object.llen_content(face_user_key) <= 0:
            r_object.srem_content(conf.redis_face_info_key_set, face_user_key)
            return
        user_id = face_user_key.split('-')[1]
        oss_running_file = "face_cluster_data/{}/.running".format(user_id)
        exist = oss_bucket.object_exists(oss_running_file)
        if exist:
            if r_object.llen_content(face_user_key) > 0:
                r_object.lpush_content(conf.redis_face_info_key_list, face_user_key)
                return
            else:
                r_object.srem_content(conf.redis_face_info_key_set, face_user_key)
                return
        oss_bucket.put_object(oss_running_file, 'running')
        r_object.srem_content(conf.redis_face_info_key_set, face_user_key)
        try:
            face_data = []
            success_image_set = set()
            while True:
                data_ = r_object.rpop_content(face_user_key)
                if not data_:
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
                oss_bucket.get_set_object(oss_suc_img_list_file, set())
                oss_face_id_with_label_file = "face_cluster_data/{}/face_id_with_label.pkl".format(user_id)
                oss_bucket.get_set_object(oss_face_id_with_label_file, dict())
                oss_face_data_file = "face_cluster_data/{}/face_data.pkl".format(user_id)
                oss_bucket.get_set_object(oss_face_data_file, list())

                suc_parser_img_set = pickle.loads(oss_bucket.get_object(oss_suc_img_list_file).read())
                face_id_label_dict = pickle.loads(oss_bucket.get_object(oss_face_id_with_label_file).read())
                old_data = pickle.loads(oss_bucket.get_object(oss_face_data_file).read())

                face_data = old_data + face_data
                start_cluster_time = time.time()
                call_res_dict, face_id_label_dict = face_cluster_interface.cluster_face_func(
                    face_data, user_id, face_id_label_dict)
                logging.info('用户ID: {}, 聚类人脸耗时: {}'.format(user_id, time.time() - start_cluster_time))

                # 回调下载成功列表
                logging.info(
                    "用户ID: {}, 开始回调 {} 结果, 共 {} 条数据...".format(
                        user_id, conf.handle_result_url, len(call_res_dict)))
                call_results_status = call_url_func(user_id, conf.handle_result_url, data_json={
                    'userId': user_id,
                    'content': call_res_dict
                })
                if not call_results_status:
                    # 回调失败, 保存结果
                    logging.error("用户ID: {}, 结果列表回调失败, 保存结果至oss!".format(user_id))
                    oss_bucket.put_object(
                        "face_cluster_call_error_data/{}/call_result_error.pkl".format(user_id),
                        pickle.dumps({
                            "handle_result_url": conf.handle_result_url,
                            "user_id": user_id,
                            "call_res_dict": call_res_dict,
                        }))
                oss_bucket.put_object(oss_face_id_with_label_file, pickle.dumps(face_id_label_dict))
                oss_bucket.put_object(oss_face_data_file, pickle.dumps(face_data))
                oss_bucket.put_object(oss_suc_img_list_file, pickle.dumps(success_image_set | suc_parser_img_set))
        except Exception as e:
            self.log_exception(e)
        finally:
            exist = oss_bucket.object_exists(oss_running_file)
            if exist:
                oss_bucket.delete_object(oss_running_file)
            if r_object.llen_content(face_user_key) > 0:
                r_set_code = r_object.sadd_content(conf.redis_face_info_key_set, face_user_key)
                if r_set_code == 1:
                    r_object.lpush_content(conf.redis_face_info_key_list, face_user_key)
            return

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        fr_arcface = face_recognition_interface.FaceRecognitionWithArcFace()
        fe_detection = face_emotion_interface.FaceEmotionKeras()  # 表情检测模型, 不能跨线程
        self.log_info("人脸聚类线程已启动...")
        while True:
            try:
                self.main_func(fe_detection, fr_arcface)
            except Exception as e:
                self.log_exception(e)
                continue
            finally:
                check_restart(1)


def get_redis_next_data(rds_name):
    params_data = r_object.rpop_content(rds_name)
    if params_data:
        params = json.loads(params_data)
        logging.info(params)
        image_url = params.get("image_url")
        res_data = image_tools.download_image(image_url, conf.tmp_image_dir)
        download_code = res_data.get('code')
        image_path = res_data.get('image_path')
        if download_code == -1:  # 下载失败, 打回列表
            reg_count = params.get('reg_count', 0)
            if reg_count > 100:
                r_object.lpush_content(
                    conf.redis_image_making_error_name,
                    json.dumps({'error_type': "download_fail", "error_data": params})
                )
                return None
            params['reg_count'] = reg_count + 1
            time.sleep(2)
            r_object.rpush_content(conf.redis_image_making_list_name, json.dumps(params))
        elif download_code == -2:  # 未知错误
            img_type = res_data.get('img_type')
            oss_key = "error_image/{}".format(os.path.basename(image_path))
            r_object.lpush_content(
                conf.redis_image_making_error_name,
                json.dumps({'error_code': -2, "img_type": img_type, "error_data": params, "oss_key": oss_key}))
            oss_bucket.put_object_from_file(oss_key, image_path)
            util.removefile(image_path)
        else:  # 图片处理成功
            params['image_path'] = image_path
            return params
    return None


class ImageProcessingThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, thread_name):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.log_content = "线程名: {}".format(thread_name) + ", {}"
        # time.sleep(0.01)

    def log_error(self, content):
        logging.error(self.log_content.format(content))

    def log_info(self, content):
        logging.info(self.log_content.format(content))

    def log_exception(self, content):
        logging.exception(self.log_content.format(content))

    def call_url_func(self, callback_url, data_json):
        call_count = 0
        call_suc_status = False
        while call_count < 20:
            try:
                call_res = requests.post(callback_url, json=data_json)
                if int(call_res.status_code) == 200:
                    call_suc_status = True
                    # self.log_info("call_status: {}".format(call_res.text))
                    return call_suc_status
                else:
                    self.log_error("call_status: {}".format(call_res.text))
                    time.sleep(9)
                    call_count += 1
            except Exception as e:
                logging.exception(e)
                call_count += 1
                time.sleep(9)
                continue
        return call_suc_status

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
                # self.log_info('renlianrenlian{}'.format(fd_scores_[0][idx]))
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
                oss_bucket.put_object_from_file(oss_face_image_name, face_image_path)
                util.removefile(face_image_path)

                # emotion_label_arg = a.detection_emotion(warped)
                # warped = np.transpose(warped, (2, 0, 1))
                # emb = b.get_feature(warped)

                redis_user_key = conf.redis_face_info_name.format(user_id)
                r_set_code = r_object.sadd_content(conf.redis_face_info_key_set, redis_user_key)
                if r_set_code == 1:
                    r_object.lpush_content(conf.redis_face_info_key_list, redis_user_key)

                r_object.lpush_content(redis_user_key, json.dumps({
                    "media_id": media_id,
                    "face_id": "{}_{}".format(media_id, idx),
                    "face_box": [left, top, right - left, bottom - top],
                    # "user_id": user_id,
                    "face_data": warped.tolist(),
                    # "face_feature": np.array(emb).tolist(),
                    # "emotionStr": emotion_label_arg,
                }))
                # self.log_info("{}_{}".format(media_id, idx))
                face_count += 1

                #
                # warped = np.transpose(warped, (2, 0, 1))
                # emb = fr_model.get_feature(warped)
                #
                # redis_user_key = conf.redis_face_info_name.format(user_id)
                # r_object.lpush_content(redis_user_key, json.dumps({
                #     "face_id": "{}_{}".format(media_id, idx),
                #     "face_box": [left, top, right - left, bottom - top],
                #     "face_feature": np.array(emb).tolist(),
                #     "emotionStr": emotion_label_arg,
                # }))
                # r_object.lpush_content(conf.redis_face_info_key_list, redis_user_key)
            # self.log_info("{}人脸耗时: {}".format(media_id, time.time() - tmp_time))
        except Exception as e:
            self.log_exception("{}上传失败\n{}".format(media_id, e))
        finally:
            return 0

    def check_restart(self, sleep_time):
        reboot_code = r_object.get_content(local_ip)
        if reboot_code == '1':
            self.log_info('发现服务需要重启, 重启代码: {}'.format(reboot_code))
            raise SystemExit
        time.sleep(sleep_time)

    def get_is_local_color(self, image):
        res = 0
        tiaojian = False
        try:
            f = min((4096. / max(image.shape[0], image.shape[1])), 1.0)
            image_r = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))
            image_np_expanded = np.expand_dims(image_r, axis=0)
            od_boxes, od_scores, od_classes = object_detection_model.detect_object(image_np_expanded)
            for idx in range(od_boxes[0].shape[0]):
                if od_scores[0][idx] < 0.7:
                    break
                if int(od_classes[0][idx]) != 1:
                    continue
                res += 1
                ymin, xmin, ymax, xmax = od_boxes[0][idx]
                if (abs((xmax - xmin) / 2. + xmin - 0.5) < 0.1) and (ymin < 0.4) and (ymax > 0.6):
                    tiaojian = True

            # scale = 600. / max(image.shape)
            # image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            # humans = pose_estimator_model.inference(image, resize_to_default=False, upsample_size=8.0)
            # for human in humans:
            #     tmp = False
            #     for k, v in human.body_parts.items():
            #         if k in [0, 1]:
            #             x_, y_ = v.x, v.y
            #             if abs(x_ - 0.5) < 0.1:
            #                 tmp = True
            #     if tmp:
            #         res += 1
            #     if res > 1:
            #         res = 0
            #         break
            # self.log_info("目标检测耗时: {}".format(time.time() - tmp_time))
        except Exception as e:
            self.log_exception(e)
        finally:
            if res == 1 and tiaojian:
                return 1
            else:
                return 0

    def main_func(self):
        start_time = time.time()
        # params_data = get_rds_next_data(conf.res_image_making_name, 3)
        params_data = get_redis_next_data(conf.redis_image_making_list_name)
        if params_data is None:
            return
        image_path = params_data.get('image_path')
        media_id = params_data.get('media_id')
        user_id = params_data.get('user_id')
        image_url = params_data.get('image_url')
        self.log_info("{} 开始处理, 还剩{}条数据".format(
            os.path.basename(image_url), r_object.llen_content(conf.redis_image_making_list_name)))

        time_dl = time.time() - start_time
        # 开始图片打标
        tmp_time = time.time()
        tag = oi_5000_model.get_tag_from_one(image_path)
        time_making = time.time() - tmp_time
        # 读取图片
        image = cv2.imread(image_path)
        # 图片证件识别
        tmp_time = time.time()
        is_card = is_idcard_model.get_res_from_one(image)
        time_ic = time.time() - tmp_time
        tmp_time = time.time()
        face_count = self.parser_face(user_id, media_id, image)
        time_face = time.time() - tmp_time
        tmp_time = time.time()
        is_local_color = self.get_is_local_color(image)
        time_od = time.time() - tmp_time

        b, g, r = cv2.split(image)
        is_black_and_white = 1 if ((b == g).all() and (b == r).all()) else 0
        data_json = {
            'mediaId': media_id,
            'fileId': params_data.get('file_id'),
            'tag': str(tag),
            'filePath': image_url,
            'exponent': detection_blur(image),
            'mediaInfo': str(json.dumps({
                "certificateInfo": is_card,
            }, ensure_ascii=False)),
            # "isBlackAndWhite": tags_list[idx].get("is_black_and_white"),
            "isBlackAndWhite": is_black_and_white,
            "isLocalColor": is_local_color,
            'existFace': min(face_count, 127),
        }
        call_results_status = self.call_url_func(params_data.get('callback_url'), data_json=data_json)

        if is_black_and_white == 1:  # 是黑白图片
            oss_key = "wonderful_tmp_dir/{}".format(os.path.basename(image_path))
            oss_bucket.put_object_from_file(oss_key, image_path)
            r_object.lpush_content(conf.redis_wonderful_gen_name, json.dumps({
                "type": 12,
                "userId": user_id,
                "mediaId": media_id,
                "oss_key": oss_key
            }))
        util.removefile(image_path)

        r_object.srem_content(conf.redis_image_making_set_name, media_id)
        self.log_info("{} total time: {}, dl time: {}, making time: {}, ic time: {}, face time: {}, od_time: {}".format(
            os.path.basename(image_url), time.time() - start_time, time_dl, time_making, time_ic, time_face, time_od))

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        logging.info("图片解析线程已启动...")
        while True:
            try:
                self.main_func()
            except Exception as e:
                self.log_exception(e)
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
        self.log_content = "线程名: {}".format(thread_name) + ", {}"
        time.sleep(0.01)

    def log_error(self, content):
        logging.error(self.log_content.format(content))

    def log_info(self, content):
        logging.info(self.log_content.format(content))

    def log_exception(self, content):
        logging.exception(self.log_content.format(content))

    def check_restart(self, params_count):
        reboot_code = r_object.get_content(local_ip)
        if reboot_code == '1':
            self.log_info('发现服务需要重启, 重启代码: {}'.format(reboot_code))
            raise SystemExit
        time.sleep(9 / max(1, params_count))

    def download_image(self, image_url):
        image_name = os.path.basename(image_url)
        image_path = os.path.join(conf.tmp_image_dir, image_name)
        try:
            urlretrieve(image_url, image_path)
        except Exception as e:
            self.log_exception("{}下载失败\n{}".format(image_url, e))
            return None
        image_id, image_type = os.path.splitext(image_name)
        if image_type.lower() == '.heic':
            try:
                new_img_path = os.path.join(conf.tmp_image_dir, "{}.jpg".format(image_id))
                tmp_time = time.time()
                image_tools.heic2jpg(image_path, new_img_path)
                self.log_info("{}转jpg耗时: {}".format(image_name, time.time() - tmp_time))
                if os.path.isfile(new_img_path):
                    util.removefile(image_path)
                    return new_img_path
                else:
                    self.log_exception("{}转换失败".format(image_url))
                    return None
            except Exception as e:
                self.log_exception("{}转换失败\n{}".format(image_url, e))
                return None
        else:
            return image_path

    def run(self):
        self.log_info("精彩生成线程已启动...")
        while True:
            try:
                params_count = r_object.llen_content(conf.redis_wonderful_gen_name)
                params = r_object.rpop_content(conf.redis_wonderful_gen_name)
                if not params:
                    self.check_restart(params_count)
                    continue
                params = json.loads(params)
                self.log_info("开始生成精彩, 剩余数据: {} 条, 当前数据: {}".format(params_count - 1, params))
                wonderful_type = params.get("type")
                # if int(wonderful_type) != 12:
                #     continue
                user_id = params.get("userId")
                # if str(user_id) != "11380 ":
                #     continue
                media_id = params.get("mediaId")
                image_url = params.get('imageUrl', None)
                image_local_path = params.get('imageLocalPath', None)
                callback_url = params.get('callbackUrl', conf.wonderful_callback_url)
                if image_url is not None:
                    image_path = self.download_image(image_url)
                else:
                    oss_key = params.get("oss_key")
                    image_path = os.path.join(conf.tmp_image_dir, os.path.basename(oss_key))
                    oss_bucket.get_object_to_file(oss_key, image_path)
                    oss_bucket.delete_object(oss_key)
                assert os.path.isfile(image_path)

                output_path = os.path.join(conf.tmp_image_dir, "{}_{}.jpg".format(media_id, wonderful_type))
                oss_image_path = "wonderful_image/{}/{}/{}_{}.jpg".format(
                    wonderful_type, user_id, media_id, wonderful_type)

                tmp_time = time.time()
                if int(wonderful_type) == 11:  # 风格化照片
                    image = imageio.imread(image_path)
                    scale = min(1920. / max(image.shape), 1.)
                    image = np.array(
                        Image.fromarray(image).resize((int(image.shape[1] * scale), int(image.shape[0] * scale))))
                    image = np.reshape(image, [1, image.shape[0], image.shape[1], 3]) / 255
                    output = image_enhancement_model.get_image(image)
                    imageio.imwrite(output_path, output[0] * 255)
                elif int(wonderful_type) == 12:  # 自动上色
                    autocolor_model.get_result_image(image_path, output_path)
                    # colorizer_model.get_result_path(image_path, output_path, render_factor=30)
                elif int(wonderful_type) == 9:  # 局部彩色
                    create_local_color.get_result(image_path, output_path)
                self.log_info("wonderful_type: {}, 耗时: {}".format(wonderful_type, time.time() - tmp_time))
                oss_bucket.put_object_from_file(oss_image_path, output_path)
                util.removefile(image_path)
                util.removefile(output_path)
                if int(wonderful_type) == 12:
                    continue
                call_url_func(user_id, callback_url, data_json={
                    "ossKey": oss_image_path,
                    "type": wonderful_type,
                    "oldMediaId": media_id,
                    "imageLocalPath": image_local_path,
                    "userId": user_id
                })

            except Exception as e:
                self.log_exception(e)
                self.log_error("风格化图片失败!")
            finally:
                self.check_restart(9)


if __name__ == '__main__':
    # 创建日志文件
    util.makedirs(conf.log_dir)
    util.makedirs(conf.tmp_image_dir)
    py_file_name = os.path.basename(__file__).split('.')[0]
    log_file = os.path.join(conf.log_dir, "{}.log".format(py_file_name))
    util.init_logging(log_file, log_filelevel=logging.INFO, log_streamlevel=logging.INFO, daily=False)
    logging.info("开始启动程序...")
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    local_ip = util.get_local_ip()  # 获取ip
    env_code = conf.env_manage.get(local_ip, 0)
    logging.info("获取运行环境代码为: {}(0: 测试服, 1: 正式服), 开始连接oss和redis...".format(env_code))
    account_conf = conf.AccountConf(env_code=env_code)
    # 连接oss
    oss_bucket = connects.ConnectALiYunOSS(
        account_conf.accessKeyId, account_conf.accessKeySecret, account_conf.endpoint, account_conf.bucket)
    # 连接redis
    r_object = connects.ConnectRedis(
        account_conf.res_host, account_conf.res_port, account_conf.res_decode_responses, account_conf.res_password)
    # 设置脚本重启代码
    r_object.set_content(local_ip, "0")

    logging.info("加载全局模型...")
    if conf.image_process_thread_num > 0:
        oi_5000_model = image_making_interface.ImageMakingWithOpenImage()
        fd_ssd_detection = face_detection_interface.FaceDetectionWithSSDMobilenet()
        fd_mtcnn_detection = face_detection_interface.FaceDetectionWithMtcnnTF(steps_threshold=[0.6, 0.7, 0.8])
        is_idcard_model = zhouwen_image_card_classify_interface.IDCardClassify()
        object_detection_model = object_detection_interface.ObjectDetectionWithSSDMobilenetV2()

    if conf.wonderful_gen_thread_num > 0:
        autocolor_model = ImageAutoColor()
        image_enhancement_model = image_enhancement_interface.AIChallengeWithDPEDSRCNN()
        pose_estimator_model = TfPoseEstimator('./models/pose_estimator_models.pb', target_size=(432, 368))
        create_local_color = ImageLocalColor()
    # fr_arcface = face_recognition_interface.FaceRecognitionWithArcFace()

    # colorizer_model = get_image_colorizer(artistic=True)
    # object_mask_detection_model = ObjectMaskDetection()

    # od_model = object_detection_interface.ObjectDetectionWithSSDMobilenetV2()

    # 创建线程并开始图片打标线程
    for i in range(conf.image_process_thread_num):
        ImageProcessingThread("Thread_{}".format(i)).start()

    # 创建线程并开始人脸聚类线程
    for i in range(conf.face_cluster_thread_num):
        FaceClusterThread("face_cluster_{}".format(i)).start()

    # 创建并开始精彩生成线程
    for i in range(conf.wonderful_gen_thread_num):
        GenerationWonderfulImageThread("wonderful_generation_{}".format(i)).start()

    while True:
        active_thread_count = threading.active_count()
        if active_thread_count == 1:
            logging.info('所有线程已停止, 等待重启...')
            break
        time.sleep(3)
