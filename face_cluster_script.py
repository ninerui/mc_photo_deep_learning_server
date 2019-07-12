import os
import time
import json
import pickle
import shutil
import logging
import datetime
import threading
from urllib.request import urlretrieve

import cv2
import mxnet as mx
import requests
import numpy as np
from tensorflow import keras

import conf
from utils import connects, util, image_tools
from dl_module import face_emotion_interface
from dl_module import face_cluster_interface
from dl_module import face_detection_interface
from dl_module import face_recognition_interface


def get_params_for_res():
    try:
        params = r_object.rpop_content(conf.res_face_cluster_name)
        if params:
            return json.loads(params)
        else:
            return None
    except Exception as e:
        logging.exception(e)
        return None


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


class FilePathConf:
    def __init__(self, user_id):
        self.oss_running_file = "face_cluster_data/{}/.running".format(user_id)
        self.oss_suc_img_list_file = "face_cluster_data/{}/suc_img_list.pkl".format(user_id)
        oss_bucket.get_set_object(self.oss_suc_img_list_file, [])
        self.oss_face_id_with_label_file = "face_cluster_data/{}/face_id_with_label.pkl".format(user_id)
        oss_bucket.get_set_object(self.oss_face_id_with_label_file, {})
        self.oss_face_data_file = "face_cluster_data/{}/face_data.pkl".format(user_id)
        oss_bucket.get_set_object(self.oss_face_data_file, [])

        self.user_path = os.path.join(conf.output_dir, str(user_id))
        util.makedirs(self.user_path)
        subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
        self.img_save_path = os.path.join(self.user_path, 'save_images_{}'.format(subdir))
        util.makedirs(self.img_save_path)
        self.face_image_path = os.path.join(self.user_path, "face_images_{}".format(subdir))
        util.makedirs(self.face_image_path)


def download_image_from_url(content, base_url, img_save_path, success_parser_image_list):
    success_img_id = []
    success_img_path = []
    try:
        for img in content:
            if img in success_parser_image_list:
                success_img_id.append(img)
                continue
            img_url = os.path.join(base_url, img)
            img_path = os.path.join(img_save_path, img)
            if os.path.isfile(img_path):
                continue
            try:
                urlretrieve(img_url, img_path)
                success_img_id.append(img)
                success_img_path.append(img_path)
            except Exception as e:
                logging.exception(e)
                logging.error("图片下载失败: {}".format(img_url))
                continue
    except Exception as e:
        logging.exception(e)
    finally:
        return success_img_id, success_img_path


def find_and_encoding_face(user_id, fr_model, fe_model, image_path_list, face_image_path):
    face_data = []
    new_success_parser_image = []
    all_count = len(image_path_list)
    conut_gl = 0
    print_time = time.time()
    for image_path in image_path_list:
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.error('image is None: {}'.format(image_path))
                oss_bucket.put_object_from_file(
                    'face_cluster_error_image/{}/{}'.format(user_id, os.path.basename(image_path)), image_path)
                continue
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_height, im_width = image.shape[:2]
            f = min((4096. / max(image.shape[0], image.shape[1])), 1.0)
            image_r = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))

            # image_np = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_r, axis=0)

            (fd_boxes_, fd_scores_) = fd_ssd_detection.detect_face(image_np_expanded)
            filename_base, _ = os.path.splitext(os.path.basename(image_path))
            for i in range(fd_boxes_[0].shape[0]):
                if fd_scores_[0][i] < 0.7:
                    break
                ymin, xmin, ymax, xmax = fd_boxes_[0][i]
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
                face_image_name = os.path.join(
                    face_image_path, "{}_{}.jpg".format(filename_base, i))
                cv2.imwrite(face_image_name, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                oss_face_image_name = "face_cluster_data/{}/face_images/{}_{}.jpg".format(user_id, filename_base, i)
                oss_bucket.put_object_from_file(oss_face_image_name, face_image_name)

                emotion_label_arg = face_emotion_interface.detection_emotion(fe_model, warped)

                warped = np.transpose(warped, (2, 0, 1))
                emb = fr_model.get_feature(warped)

                face_data.append({
                    "face_id": "{}_{}".format(filename_base, i),
                    "face_box": [left, top, right - left, bottom - top],
                    "face_feature": emb,
                    "emotionStr": emotion_label_arg,
                })
        except Exception as e:
            logging.exception(e)
            logging.error('image is error: {}'.format(image_path))
            oss_bucket.put_object_from_file(
                'face_cluster_error_image/{}/{}'.format(user_id, os.path.basename(image_path)), image_path)
            continue
        finally:
            if conut_gl % 100 == 0:
                logging.info(
                    "用户ID: {}, 进度: {}/{}, 100张耗时: {}".format(user_id, conut_gl, all_count, time.time() - print_time))
                print_time = time.time()
            conut_gl += 1
            new_success_parser_image.append(os.path.basename(image_path))
    return face_data, new_success_parser_image


def face_cluster_main():
    thread_id = threading.current_thread().getName()
    try:
        logging.info("线程ID: {}, 开始加载模型...".format(thread_id))
        fr_arcface = face_recognition_interface.FaceRecognitionWithArcFace(ctx, model_path_with_arcface)
        emotion_classifier = keras.models.load_model(model_path_with_emotion, compile=False)  # 表情检测模型, 不能跨线程
        logging.info("线程ID: {}, 模型加载成功".format(thread_id))
    except Exception as e:
        logging.exception(e)
        return
    while True:
        start_time = time.time()
        params_count = r_object.llen_content(conf.res_face_cluster_name)
        logging.info("线程ID: {}, 从 redis 获取数据, 还剩 {} 条!".format(thread_id, params_count))
        params = get_params_for_res()
        if params:
            user_id = params['user_id']
            base_url = params['base_url']
            content = params['content']
            callback_url = params['callback_url']
            handle_result_url = params['handle_result_url']
            file_conf = FilePathConf(user_id)
            call_res_dict = []
            try:
                oss_bucket.put_object(file_conf.oss_running_file, 'running')
                # 获取成功处理过的文件列表
                suc_parser_img_list = pickle.loads(oss_bucket.get_object(file_conf.oss_suc_img_list_file).read())
                # 下载图片
                logging.info("线程ID: {}, 用户ID: {}, 开始下载图片...".format(thread_id, user_id))

                start_download_time = time.time()
                success_img_id, success_img_path = download_image_from_url(
                    content, base_url, file_conf.img_save_path, suc_parser_img_list)
                logging.info("线程ID: {}, 用户ID: {}, 下载图片完成 {}/{}, 下载人脸耗时: {}".format(
                    thread_id, user_id, len(success_img_id), len(content), time.time() - start_download_time))

                # 回调下载成功列表
                call_download_status = call_url_func(user_id, callback_url, data_json={
                    'userId': user_id,
                    'baseUrl': base_url,
                    'content': success_img_id
                })

                if not call_download_status:
                    # 回调失败, 保存结果
                    logging.error("线程ID: {}, 用户ID: {}, 下载列表回调失败, 保存结果至oss!".format(thread_id, user_id))
                    oss_bucket.put_object(
                        "face_cluster_call_error_data/{}/call_download_error.pkl".format(user_id),
                        pickle.dumps({
                            "callback_url": callback_url,
                            "user_id": user_id,
                            "base_url": base_url,
                            "success_img_id": success_img_id,
                            "success_img_path": success_img_path,
                        }))
                del success_img_id  # 删除回调的下载成功列表
                logging.info("线程ID: {}, 用户ID: {}, 开始分析图片...".format(thread_id, user_id))

                start_parser_time = time.time()
                face_data, new_success_parser_image = find_and_encoding_face(
                    user_id, fr_arcface, emotion_classifier, success_img_path, file_conf.face_image_path)
                logging.info('线程ID: {}, 用户ID: {}, 处理图片数: {}/{}, 处理图片耗时: {}'.format(
                    thread_id, user_id, len(new_success_parser_image),
                    len(success_img_path), time.time() - start_parser_time))

                oss_bucket.put_object(
                    file_conf.oss_suc_img_list_file, pickle.dumps(suc_parser_img_list + new_success_parser_image))

                logging.info("线程ID: {}, 用户ID: {}, 开始聚类人脸...".format(thread_id, user_id))
                if len(face_data) > 0:
                    # 获取已聚类的列表
                    face_id_label_dict = pickle.loads(
                        oss_bucket.get_object(file_conf.oss_face_id_with_label_file).read())
                    # 获取原先的face data, 组合成新的face data拿去聚类
                    old_data = pickle.loads(oss_bucket.get_object(file_conf.oss_face_data_file).read())
                    face_data = old_data + face_data

                    # 开始聚类人脸
                    start_cluster_time = time.time()
                    call_res_dict, face_id_label_dict = face_cluster_interface.cluster_face_func(
                        face_data, user_id, face_id_label_dict)
                    logging.info('线程ID: {}, 用户ID: {}, 聚类人脸耗时: {}'.format(
                        thread_id, user_id, time.time() - start_cluster_time))

                    oss_bucket.put_object(file_conf.oss_face_id_with_label_file, pickle.dumps(face_id_label_dict))
                    oss_bucket.put_object(file_conf.oss_face_data_file, pickle.dumps(face_data))
            except Exception as e:
                logging.exception(e)
            finally:
                if os.path.exists(file_conf.img_save_path):  # 删除本地的原始图片
                    shutil.rmtree(file_conf.img_save_path)
                if os.path.exists(file_conf.face_image_path):  # 删除本地创建的人脸图片
                    shutil.rmtree(file_conf.face_image_path)
                # 判断oss .running文件并删除
                exist = oss_bucket.object_exists(file_conf.oss_running_file)
                if exist:
                    oss_bucket.delete_object(file_conf.oss_running_file)
                # 开始回调结果数据
                try:
                    # 回调下载成功列表
                    logging.info(
                        "线程ID: {}, 用户ID: {}, 开始回调 {} 结果, 共 {} 条数据...".format(
                            thread_id, user_id, handle_result_url, len(call_res_dict)))
                    call_results_status = call_url_func(user_id, handle_result_url, data_json={
                        'userId': user_id,
                        'content': call_res_dict
                    })

                    if not call_results_status:
                        # 回调失败, 保存结果
                        logging.error("线程ID: {}, 用户ID: {}, 结果列表回调失败, 保存结果至oss!".format(thread_id, user_id))
                        oss_bucket.put_object(
                            "face_cluster_call_error_data/{}/call_result_error.pkl".format(user_id),
                            pickle.dumps({
                                "handle_result_url": handle_result_url,
                                "user_id": user_id,
                                "call_res_dict": call_res_dict,
                            }))
                    logging.info('线程ID: {}, 用户ID: {}, 整个流程耗时: {}'.format(
                        thread_id, user_id, time.time() - start_time))
                except Exception as e:
                    logging.exception(e)
        else:
            logging.info("线程ID: {}, 暂时没有数据获取, 进入等待中(每等待9s进行检查)...".format(thread_id))
            while True:
                ret_code = wait_restart_server(thread_id)
                if ret_code == 1:
                    return
                params_count = r_object.llen_content(conf.res_face_cluster_name)
                if params_count >= 1:
                    logging.info("线程ID: {}, 检测到任务, 跳出等待".format(thread_id))
                    break
                time.sleep(9)
        ret_code = wait_restart_server(thread_id)
        if ret_code == 1:
            return


def wait_restart_server(current_thread_name):
    reboot_status = r_object.get_content(local_ip)
    reboot_code = str(reboot_status)
    if reboot_code == '1':
        logging.info('线程ID: {}, 发现服务需要重启, 重启代码: {}'.format(current_thread_name, reboot_code))
        return 1
    return


def main():
    # 创建线程
    thread_list = [threading.Thread(target=face_cluster_main, name=str(i)) for i in range(thread_num)]
    # 启动线程
    for i in thread_list:
        i.start()
    while True:
        thread_status = [i.is_alive() for i in thread_list]
        if sum(thread_status) == 0:
            logging.info('所有线程已停止, 等待重启...')
            return
        time.sleep(20)


if __name__ == '__main__':
    # 创建日志
    util.makedirs("./log")
    log_file = "./log/face_cluster_script.log"
    util.init_logging(log_file, log_filelevel=logging.INFO, log_streamlevel=logging.INFO, daily=False)
    start_sleep_time = 9
    logging.info("等待 {}s 之后启动程序...".format(start_sleep_time))
    time.sleep(start_sleep_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    local_ip = util.get_local_ip()
    env_code = conf.env_manage.get(local_ip, 0)
    logging.info("获取到的脚本运行环境代码为: {}".format(env_code))
    account_conf = conf.AccountConf(env_code=env_code)
    # 连接oss
    oss_bucket = connects.ConnectALiYunOSS(
        account_conf.accessKeyId, account_conf.accessKeySecret, account_conf.endpoint, account_conf.bucket)
    # 连接redis
    r_object = connects.ConnectRedis(
        account_conf.res_host, account_conf.res_port, account_conf.res_decode_responses, account_conf.res_password)
    # 获取本地ip
    r_object.set_content(local_ip, "0")
    ctx = mx.cpu()
    model_path_with_arcface = './models/fr_model_r100_ii/model'
    model_path_with_emotion = './models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    fd_ssd_detection = face_detection_interface.FaceDetectionWithSSDMobilenet()
    fd_mtcnn_detection = face_detection_interface.FaceDetectionWithMtcnnTF(steps_threshold=[0.6, 0.7, 0.8])

    thread_num = 3
    logging.info("即将开启的线程数: {}".format(thread_num))

    main()
